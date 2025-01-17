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
from loss import TverskyLoss


class Classifier(LightningModule):
    def __init__(self, lr_dino, lr_class, weight_decay, k):
        super(Classifier, self).__init__()
        self.lr_dino = lr_dino
        self.lr_class = lr_class
        self.weight_decay = weight_decay
        self.k = k

        self.tverLoss = TverskyLoss()
        self.ceLoss_ppgl = nn.CrossEntropyLoss(
            weight=350 / torch.Tensor([99.0, 362.0, 114.0, 75.0])
        )
        self.ceLoss_reg = nn.CrossEntropyLoss(
            weight=350 / torch.Tensor([305.0, 201.0, 80.0, 64.0])
        )

        self.setup_metrics()

        backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()
        vitconf = ViTAdapterConfig()
        self.backbone = ViTFeatureExtractor(backbone, vitconf)

        self.feature_extractor = nn.Sequential(
            nn.Linear(384, 256),
            nn.LeakyReLU(),
        )

        self.reg_predictor = nn.Linear(256, 4)
        
        self.classifiers = nn.ModuleList()
        for _ in range(4):
            self.classifiers.append(
                nn.Sequential(
                    nn.Linear(256, 256, False),
                    nn.BatchNorm1d(256),
                    nn.LeakyReLU(),
                    nn.Dropout1d(),
                    nn.Linear(256, 4),
                )
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
            self.reg_predictor.modules(),
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
        reg_pred = self.reg_predictor(img_feat)

        out = torch.zeros((bsz, 4, 4)).cuda()
        for idx, classifier in enumerate(self.classifiers):
            out[:, idx, ...] += classifier(img_feat)

        if train:
            out *= reg.view(bsz, 4, 1)
        else:
            out *= self.softmax(reg_pred).view(bsz, 4, 1)

        out = out.sum(dim=-2)

        return out, reg_pred

    def training_step(self, batch, batch_idx):
        reg, y, img, _ = batch

        outputs, reg_pred = self.forward(img, reg)

        opt_dino, opt_features, opt_ppgl_classifier, opt_reg = self.optimizers()

        opt_dino.zero_grad()
        opt_ppgl_classifier.zero_grad()
        opt_features.zero_grad()
        opt_reg.zero_grad()
        loss = (
            self.ceLoss_ppgl(outputs, torch.argmax(y, dim=-1))
            + self.ceLoss_reg(reg_pred, torch.argmax(reg, dim=-1)),
            +self.tverLoss(outputs, y),
        )
        self.manual_backward(loss)
        opt_dino.step()
        opt_ppgl_classifier.step()
        opt_features.step()
        opt_reg.step()

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

        outputs, reg_pred = self.forward(img, reg, False)

        acc_multiclass = self.multiclass_accuracy(outputs, torch.argmax(y, dim=-1))
        acc_total = self.total_accuracy(outputs, torch.argmax(y, dim=-1))

        log_dict = {
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

        return {"outputs": outputs}

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

        outputs, reg_pred = self.forward(img, reg, False)

        acc_multiclass = self.multiclass_accuracy(outputs, torch.argmax(y, dim=-1))
        acc_total = self.total_accuracy(outputs, torch.argmax(y, dim=-1))

        f1_multiclass = self.multiclass_f1(outputs, torch.argmax(y, dim=-1))
        f1_total = self.total_f1(outputs, torch.argmax(y, dim=-1))

        auc_multiclass = self.multiclass_auc(outputs, torch.argmax(y, dim=-1))
        auc_total = self.total_auc(outputs, torch.argmax(y, dim=-1))

        rec_multiclass = self.multiclass_rec(outputs, torch.argmax(y, dim=-1))
        rec_total = self.total_rec(outputs, torch.argmax(y, dim=-1))

        log_dict = {
            "test_f1": f1_total,
            "test_auc": auc_total,
            "test_acc": acc_total,
            "test_rec": rec_total,
        }

        for i, acc_ in enumerate(acc_multiclass):
            log_dict["test_acc_" + str(i)] = acc_

        for i, acc_ in enumerate(f1_multiclass):
            log_dict["test_f1_" + str(i)] = acc_

        for i, acc_ in enumerate(auc_multiclass):
            log_dict["test_auc_" + str(i)] = acc_

        for i, acc_ in enumerate(rec_multiclass):
            log_dict["test_rec_" + str(i)] = acc_

        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return {"outputs": outputs}

    def predict(self, x):
        return self.forward(x)

    def configure_optimizers(self):
        optimizer_dino = optim.AdamW(
            self.backbone.parameters(), lr=self.lr_dino, weight_decay=self.weight_decay
        )
        optimizer_features = optim.Adam(
            self.feature_extractor.parameters(), lr=self.lr_class
        )
        optimizer_ppgl_classifier = optim.Adam(
            self.classifiers.parameters(), lr=self.lr_class
        )
        optimizer_region_classifier = optim.Adam(
            self.reg_predictor.parameters(), lr=self.lr_class
        )

        return [
            optimizer_dino,
            optimizer_features,
            optimizer_ppgl_classifier,
            optimizer_region_classifier,
        ]
