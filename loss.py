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