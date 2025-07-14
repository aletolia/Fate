import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    It was proposed in https://arxiv.org/abs/1708.02002.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (raw logits).
            targets: Ground truth labels.
        """
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = torch.exp(-bce_loss)
        
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        focal_loss = alpha_t * (1 - p_t).pow(self.gamma) * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SymmetricCrossEntropyLoss(nn.Module):
    """
    Symmetric Cross Entropy for robust learning with noisy labels.
    It was proposed in https://arxiv.org/abs/1908.06112.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 1.0, reduction: str = 'mean'):
        super(SymmetricCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (raw logits).
            targets: Ground truth labels.
        """
        # CCE
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # RCE
        probs = torch.sigmoid(logits)
        # Clamp targets to avoid log(0)
        targets_clipped = torch.clamp(targets, 1e-7, 1.0 - 1e-7)
        rce_loss = - (probs * torch.log(targets_clipped) + (1 - probs) * torch.log(1 - targets_clipped))

        loss = self.alpha * ce_loss + self.beta * rce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
