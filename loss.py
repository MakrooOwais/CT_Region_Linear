from __future__ import print_function

import torch
import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1e-6):
        """
        Tversky Loss implementation for binary classification.
        Supports both one-dimensional and two-dimensional outputs.

        Parameters:
        alpha (float): Weight for false negatives.
        beta (float): Weight for false positives.
        smooth (float): Small constant to avoid division by zero.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        """
        Forward pass for Tversky Loss.

        Parameters:
        y_pred (torch.Tensor): Predicted probabilities (after sigmoid for binary classification).
        y_true (torch.Tensor): Ground truth binary labels.

        Returns:
        torch.Tensor: Tversky loss value.
        """
        # Ensure the tensors are of the same shape
        assert (
            y_pred.shape == y_true.shape
        ), "y_pred and y_true must have the same shape"

        # Flatten the tensors for computation
        if len(y_pred.shape) > 1:
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)

        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)

        # Compute True Positives (TP), False Positives (FP), and False Negatives (FN)
        TP = (y_true * y_pred).sum()
        FP = ((1 - y_true) * y_pred).sum()
        FN = (y_true * (1 - y_pred)).sum()

        # Compute Tversky Index
        tversky_index = (TP + self.smooth) / (
            TP + self.alpha * FN + self.beta * FP + self.smooth
        )

        # Return Tversky loss
        return 1 - tversky_index

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_mode="all", base_temperature=0.07):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute KLD-based loss for model. If both `labels` and `mask` are None,
        it degenerates to unsupervised KLD loss.

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # compute probabilities
        exp_logits = torch.exp(logits) * logits_mask
        prob_dist = exp_logits / exp_logits.sum(1, keepdim=True)

        # create the target distribution using the mask
        target_dist = mask / mask.sum(1, keepdim=True)
        target_dist = target_dist + 1e-6  # Avoid division by zero

        # compute KLD loss
        kld_loss = target_dist * (torch.log(target_dist) - torch.log(prob_dist + 1e-6))
        kld_loss = kld_loss.sum(1).mean()

        return kld_loss
