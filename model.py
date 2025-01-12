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

from model_peft import ViTAdapterConfig, ViTFeatureExtractor
from loss import SupervisedContrastiveLoss, TverskyLoss


class Classifier(LightningModule):
    def __init__(self, lr, weight_decay, k):
        super(Classifier, self).__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.k = k

        self.tverLoss = TverskyLoss()
        self.conLoss = SupervisedContrastiveLoss()
        self.ceLoss_ppgl = nn.CrossEntropyLoss(
            weight=362 / torch.Tensor([99.0, 362.0, 114.0, 75.0])
        )

        self.setup_metrics()

        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        vitconf = ViTAdapterConfig()
        self.backbone = ViTFeatureExtractor(backbone, vitconf)

        self.feature_extractor = nn.Sequential(
            nn.Linear(768, 256),
            nn.LeakyReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 4),
        )
        self.softmax = nn.Softmax(dim=-1)

        self.dim = 1
        self.eps = 1e-6
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
            self.region_classifier.modules(),
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

    def forward(self, img, reg, train=True):
        bsz = img[0].shape[0]
        img = torch.cat(img, dim=0)
        img = self.backbone(img)
        img_feat = self.feature_extractor(img)

        f1, f2 = torch.split(img_feat, [bsz, bsz], dim=0)
        features = torch.cat(
            [
                F.normalize(f1, p=2, dim=self.dim, eps=self.eps).unsqueeze(1),
                F.normalize(f2, p=2, dim=self.dim, eps=self.eps).unsqueeze(1),
            ],
            dim=1,
        )

        out = self.classifier(f1)

        return out, features

    def training_step(self, batch, batch_idx):
        reg, y, img, _ = batch

        outputs, features = self.forward(img, reg)

        opt_features, opt_ppgl_classifier = self.optimizers()

        opt_ppgl_classifier.zero_grad()
        opt_features.zero_grad()
        loss = (
            self.ceLoss_ppgl(outputs, torch.argmax(y, dim=-1))
            + self.tverLoss(outputs, y)
            + self.conLoss(features)
        )
        self.manual_backward(loss)
        opt_ppgl_classifier.step()
        opt_features.step()

        acc_multiclass = self.multiclass_accuracy(outputs, torch.argmax(y, dim=-1))
        acc_total = self.total_accuracy(outputs, torch.argmax(y, dim=-1))

        log_dict = {
            "train_loss": loss,
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

        return {"loss": loss, "outputs": outputs}

    def validation_step(self, batch, batch_idx):
        reg, y, img, _ = batch

        outputs, features = self.forward(img, reg, False)

        loss = (
            self.ceLoss_ppgl(outputs, torch.argmax(y, dim=-1))
            + self.tverLoss(outputs, y)
            + self.conLoss(features)
        )

        acc_multiclass = self.multiclass_accuracy(outputs, torch.argmax(y, dim=-1))
        acc_total = self.total_accuracy(outputs, torch.argmax(y, dim=-1))

        log_dict = {
            "val_loss": loss,
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

        return {"loss": loss, "outputs": outputs}

    def on_validation_epoch_end(self):
        not_complete = False
        accs = torch.zeros(4).cuda()
        for i in range(4):
            acc = self.trainer.callback_metrics[f"val_acc_{i}"]
            accs[i] += acc
            if acc < 0.55:
                not_complete = True

        if not not_complete:
            torch.save(self.state_dict, f"model_{self.k}.pt")
            self.trainer.should_stop = True

        res = accs.mean() - accs.std()

        if res > self.best_val:
            self.best_val = res
            torch.save(self.state_dict, f"model_{self.k}.pt")

    def on_before_test_epoch(self):
        self.load_state_dict(torch.load(f"model_{self.k}.pt"))

    def test_step(self, batch, batch_idx):
        reg, y, img, _ = batch

        outputs, features = self.forward(img, reg, False)

        loss = (
            self.ceLoss_ppgl(outputs, torch.argmax(y, dim=-1))
            + self.tverLoss(outputs, y)
            + self.conLoss(features)
        )

        acc_multiclass = self.multiclass_accuracy(outputs, torch.argmax(y, dim=-1))
        acc_total = self.total_accuracy(outputs, torch.argmax(y, dim=-1))

        f1_multiclass = self.multiclass_f1(outputs, torch.argmax(y, dim=-1))
        f1_total = self.total_f1(outputs, torch.argmax(y, dim=-1))

        auc_multiclass = self.multiclass_auc(outputs, torch.argmax(y, dim=-1))
        auc_total = self.total_auc(outputs, torch.argmax(y, dim=-1))

        rec_multiclass = self.multiclass_rec(outputs, torch.argmax(y, dim=-1))
        rec_total = self.total_rec(outputs, torch.argmax(y, dim=-1))

        log_dict = {
            "val_loss": loss,
            "val_f1": f1_total,
            "val_auc": auc_total,
            "val_acc": acc_total,
            "val_rec": rec_total,
        }

        for i, acc_ in enumerate(acc_multiclass):
            log_dict["val_acc_" + str(i)] = acc_

        for i, acc_ in enumerate(f1_multiclass):
            log_dict["val_f1_" + str(i)] = acc_

        for i, acc_ in enumerate(auc_multiclass):
            log_dict["val_auc_" + str(i)] = acc_

        for i, acc_ in enumerate(rec_multiclass):
            log_dict["val_rec_" + str(i)] = acc_

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"loss": loss, "outputs": outputs}

    def predict(self, x):
        return self.forward(x)

    def configure_optimizers(self):
        optimizer_features = optim.AdamW(
            self.feature_extractor.parameters(), lr=self.lr, weight_decay=2e-5
        )
        optimizer_ppgl_classifier = optim.Adam(self.classifier.parameters(), lr=self.lr)

        return [
            optimizer_features,
            optimizer_ppgl_classifier,
        ]
