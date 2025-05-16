import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassAUROC,
    MulticlassRecall,
)
from torchvision.models import swin_transformer
from loss import TverskyLoss


class AnatomicalLocationWeightEstimator(nn.Module):
    def __init__(self, input_dim=768):
        super().__init__()
        self.fc = nn.Linear(input_dim, 3)  # 3 anatomical locations

    def forward(self, x):
        return F.softmax(self.fc(x), dim=1)


class LocationSpecificClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
        )
        self.fc2 = nn.Linear(hidden_dim, 4)  # 4 genetic clusters

    def forward(self, x):
        x = self.fc1(x)
        return self.fc2(x)


class Classifier(LightningModule):
    def __init__(
        self,
        lr_dino=1e-5,
        lr_class=1e-2,
        weight_decay=0.0005,
        k=0,
        lambda_weight=0.1,
        alpha=0.7,
        beta=0.3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr_dino = lr_dino
        self.lr_class = lr_class
        self.weight_decay = weight_decay
        self.k = k
        self.lambda_weight = lambda_weight
        self.alpha = alpha
        self.beta = beta

        # Loss functions
        self.tversky_loss = TverskyLoss(alpha=alpha, beta=beta)
        self.ce_loss_ppgl = nn.CrossEntropyLoss(
            weight=350 / torch.Tensor([99.0, 362.0, 114.0, 75.0])
        )
        self.ce_loss_region = nn.CrossEntropyLoss(
            weight=350
            / torch.Tensor([506.0, 80.0, 64.0])  # Abdomen, Chest, Head & Neck
        )

        # Metrics
        self.setup_metrics()

        # Feature extractor (Swin Transformer backbone)
        self.backbone = swin_transformer.swin_t(weights="DEFAULT")
        self.backbone.head = nn.Identity()  # Replace classification head with identity
        self.feature_dim = 768  # Output dimension of Swin-T

        # Feature projection layer
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LeakyReLU(),
        )

        # Anatomical location weight estimator
        self.location_estimator = AnatomicalLocationWeightEstimator(256)

        # Location-specific classifiers
        self.classifiers = nn.ModuleList(
            [
                LocationSpecificClassifier(256, 256)  # Head & Neck
                for _ in range(3)  # 3 anatomical locations
            ]
        )

        self.best_val = -float("inf")
        self.init_weights()
        self.automatic_optimization = False

    def setup_metrics(self):
        """Initialize all metrics for evaluation"""
        self.multiclass_accuracy = MulticlassAccuracy(num_classes=4, average=None)
        self.total_accuracy = MulticlassAccuracy(num_classes=4)
        self.multiclass_f1 = MulticlassF1Score(num_classes=4, average=None)
        self.total_f1 = MulticlassF1Score(num_classes=4)
        self.multiclass_auc = MulticlassAUROC(num_classes=4, average=None)
        self.total_auc = MulticlassAUROC(num_classes=4)
        self.multiclass_rec = MulticlassRecall(num_classes=4, average=None)
        self.total_rec = MulticlassRecall(num_classes=4)

    def init_weights(self):
        for sub_model in [
            self.feature_extractor.modules(),
            self.classifiers.modules(),
            self.location_estimator.modules(),
        ]:
            for m in sub_model:
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def forward(self, img, region_gt=None, train=True):
        bsz = img.shape[0]

        # Extract features using the backbone
        features = self.backbone(img)

        # Project features
        features = self.feature_extractor(features)

        # Get anatomical location weights
        location_weights = self.location_estimator(features)

        # Get predictions from each location-specific classifier
        genetic_preds = torch.zeros((bsz, 3, 4)).to(img.device)
        for idx, classifier in enumerate(self.classifiers):
            genetic_preds[:, idx] = classifier(features)

        # Weighted aggregation
        if train and region_gt is not None:
            # During training, use ground truth region weights
            weighted_pred = torch.sum(genetic_preds * region_gt.view(bsz, 3, 1), dim=1)
        else:
            # During inference, use predicted region weights
            weighted_pred = torch.sum(
                genetic_preds * location_weights.view(bsz, 3, 1), dim=1
            )

        return weighted_pred, location_weights

    def training_step(self, batch, batch_idx):
        region, y, img, _ = batch

        # Convert region from one-hot to 3-class format (removing the 4th dimension which is unused)
        region = region[:, :3]

        # Forward pass
        genetic_pred, location_weights = self.forward(img, region)

        # Get optimizers
        opt_backbone, opt_features, opt_classifiers, opt_location = self.optimizers()

        # Zero gradients
        opt_backbone.zero_grad()
        opt_features.zero_grad()
        opt_classifiers.zero_grad()
        opt_location.zero_grad()

        # Calculate losses
        ce_loss = self.ce_loss_ppgl(genetic_pred, torch.argmax(y, dim=-1))
        tv_loss = self.tversky_loss(genetic_pred, y)
        ppgl_loss = ce_loss + tv_loss

        location_loss = self.ce_loss_region(
            location_weights, torch.argmax(region, dim=-1)
        )

        # Combined loss as per paper
        total_loss = ppgl_loss + self.lambda_weight * location_loss

        # Backward pass
        self.manual_backward(total_loss)

        # Update weights
        opt_backbone.step()
        opt_features.step()
        opt_classifiers.step()
        opt_location.step()

        # Log metrics
        acc_multiclass = self.multiclass_accuracy(genetic_pred, torch.argmax(y, dim=-1))
        acc_total = self.total_accuracy(genetic_pred, torch.argmax(y, dim=-1))

        log_dict = {
            "train_loss": total_loss,
            "train_ppgl_loss": ppgl_loss,
            "train_location_loss": location_loss,
            "train_acc": acc_total,
        }

        for i, acc_ in enumerate(acc_multiclass):
            log_dict["train_acc_" + str(i)] = acc_

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": total_loss, "outputs": genetic_pred}

    def validation_step(self, batch, batch_idx):
        region, y, img, _ = batch

        # Convert region from one-hot to 3-class format
        region = region[:, :3]

        # Forward pass (don't use ground truth regions during validation)
        genetic_pred, location_weights = self.forward(img, train=False)

        # Calculate losses
        ce_loss = self.ce_loss_ppgl(genetic_pred, torch.argmax(y, dim=-1))
        tv_loss = self.tversky_loss(genetic_pred, y)
        ppgl_loss = ce_loss + tv_loss

        location_loss = self.ce_loss_region(
            location_weights, torch.argmax(region, dim=-1)
        )

        # Combined loss
        total_loss = ppgl_loss + self.lambda_weight * location_loss

        # Log metrics
        acc_multiclass = self.multiclass_accuracy(genetic_pred, torch.argmax(y, dim=-1))
        acc_total = self.total_accuracy(genetic_pred, torch.argmax(y, dim=-1))

        log_dict = {
            "val_loss": total_loss,
            "val_acc": acc_total,
        }

        for i, acc_ in enumerate(acc_multiclass):
            log_dict["val_acc_" + str(i)] = acc_

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"outputs": genetic_pred}

    def on_validation_epoch_end(self):
        not_complete = False
        accs = torch.zeros(4).to(self.device)

        for i in range(4):
            acc = self.trainer.callback_metrics[f"val_acc_{i}"]
            accs[i] += acc
            if acc < 0.55:
                not_complete = True

        if not not_complete:
            torch.save(self.state_dict(), f"model_{self.k}.pt")
            self.trainer.should_stop = True

        res = accs.mean() - accs.std()
        if res > self.best_val:
            self.best_val = res
            torch.save(self.state_dict(), f"model_{self.k}.pt")

    def on_test_epoch_start(self):
        print("loaded best dict")
        self.load_state_dict(torch.load(f"model_{self.k}.pt"))

    def test_step(self, batch, batch_idx):
        region, y, img, _ = batch

        # Convert region from one-hot to 3-class format
        region = region[:, :3]

        # Forward pass
        genetic_pred, _ = self.forward(img, train=False)

        # Calculate metrics
        acc_multiclass = self.multiclass_accuracy(genetic_pred, torch.argmax(y, dim=-1))
        acc_total = self.total_accuracy(genetic_pred, torch.argmax(y, dim=-1))
        f1_multiclass = self.multiclass_f1(genetic_pred, torch.argmax(y, dim=-1))
        f1_total = self.total_f1(genetic_pred, torch.argmax(y, dim=-1))
        auc_multiclass = self.multiclass_auc(genetic_pred, torch.argmax(y, dim=-1))
        auc_total = self.total_auc(genetic_pred, torch.argmax(y, dim=-1))
        rec_multiclass = self.multiclass_rec(genetic_pred, torch.argmax(y, dim=-1))
        rec_total = self.total_rec(genetic_pred, torch.argmax(y, dim=-1))

        # Log metrics
        log_dict = {
            "test_f1": f1_total,
            "test_auc": auc_total,
            "test_acc": acc_total,
            "test_rec": rec_total,
        }

        for name, met in [
            ("acc", acc_multiclass),
            ("f1", f1_multiclass),
            ("auc", auc_multiclass),
            ("rec", rec_multiclass),
        ]:
            for i, val in enumerate(met):
                log_dict[f"test_{name}_{i}"] = val

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"outputs": genetic_pred}

    def configure_optimizers(self):
        optimizer_backbone = optim.AdamW(
            self.backbone.parameters(),
            lr=self.lr_dino,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
        )

        optimizer_features = optim.AdamW(
            self.feature_extractor.parameters(), lr=self.lr_class, betas=(0.9, 0.999)
        )

        optimizer_classifiers = optim.AdamW(
            self.classifiers.parameters(), lr=self.lr_class, betas=(0.9, 0.999)
        )

        optimizer_location = optim.AdamW(
            self.location_estimator.parameters(), lr=self.lr_class, betas=(0.9, 0.999)
        )

        return [
            optimizer_backbone,
            optimizer_features,
            optimizer_classifiers,
            optimizer_location,
        ]
