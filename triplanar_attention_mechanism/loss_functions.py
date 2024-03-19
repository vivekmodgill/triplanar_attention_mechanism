# triplanar_attention_mechanism/loss_functions.py

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryCrossEntropySorensenDiceLossFunction(nn.Module):
    """
    Binary cross-entropy Sorensen-Dice loss function module.

    Attributes:
        alpha         (float):                          Weight.
        cross_entropy (nn.BCEWithLogitsLoss):           Binary cross-entropy loss function.
        sorensen_dice (BinarySorensenDiceLossFunction): Binary Sorensen-Dice loss function.

    Args:
        alpha   (float, optional): Weight.                                           Defaults to .5.
        epsilon (float, optional): Smoothness of binary Sorensen-Dice loss function. Defaults to 1.
    """
    def __init__(self, alpha: float = .5, epsilon: float = 1) -> None:
        super().__init__()
        self.alpha = alpha
        self.cross_entropy = nn.BCEWithLogitsLoss()
        self.sorensen_dice = BinarySorensenDiceLossFunction(epsilon)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            inputs  (torch.Tensor): Logit tensor       (N, *).
            targets (torch.Tensor): Probability tensor (N, *).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Binary cross-entropy Sorensen-Dice, cross-entropy and Sorensen-Dice losses.
        """
        cross_entropy = self.cross_entropy(inputs, targets)
        sorensen_dice = self.sorensen_dice(inputs, targets)
        return self.alpha * cross_entropy + (1 - self.alpha) * sorensen_dice, cross_entropy, sorensen_dice

class BinarySorensenDiceLossFunction(nn.Module):
    """
    Binary Sorensen-Dice loss function module.

    Attributes:
        epsilon (float): Smoothness.

    Args:
        epsilon (float, optional): Smoothness. Defaults to 1.
    """
    def __init__(self, epsilon: float = 1) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs  (torch.Tensor): Logit tensor       (N, *).
            targets (torch.Tensor): Probability tensor (N, *).

        Returns:
            torch.Tensor: Binary Sorensen-Dice loss.
        """
        inputs = torch.atleast_2d(F.Sigmoid(inputs)).flatten(1)
        targets = torch.atleast_2d(targets).flatten(1)
        return 1 - ((2 * (inputs * targets).sum(1) + self.epsilon) / (inputs.sum(1) + targets.sum(1) + self.epsilon)).mean()

class MulticlassCrossEntropySorensenDiceLossFunction(nn.Module):
    """
    Multiclass cross-entropy Sorensen-Dice loss function module.

    Attributes:
        alpha         (float):                              Weight.
        cross_entropy (nn.BCEWithLogitsLoss):               Multiclass cross-entropy loss function.
        sorensen_dice (MulticlassSorensenDiceLossFunction): Multiclass Sorensen-Dice loss function.

    Args:
        alpha   (float, optional): Weight.                                               Defaults to .5.
        epsilon (float, optional): Smoothness of multiclass Sorensen-Dice loss function. Defaults to 1.
    """
    def __init__(self, alpha: float = .5, epsilon: float = 1) -> None:
        super().__init__()
        self.alpha = alpha
        self.cross_entropy = nn.CrossEntropyLoss()
        self.sorensen_dice = MulticlassSorensenDiceLossFunction(epsilon)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            inputs  (torch.Tensor): Logit tensor (N, C, *).
            targets (torch.Tensor): Index tensor (N, *).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Multiclass cross-entropy Sorensen-Dice, cross-entropy and Sorensen-Dice losses.
        """
        cross_entropy = self.cross_entropy(inputs, targets)
        sorensen_dice = self.sorensen_dice(inputs, targets)
        return self.alpha * cross_entropy + (1 - self.alpha) * sorensen_dice, cross_entropy, sorensen_dice

class MulticlassSorensenDiceLossFunction(nn.Module):
    """
    Multiclass Sorensen-Dice loss function module.

    Attributes:
        epsilon (float): Smoothness.

    Args:
        epsilon (float, optional): Smoothness. Defaults to 1.
    """
    def __init__(self, epsilon: float = 1) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            inputs  (torch.Tensor): Logit tensor (N, C, *).
            targets (torch.Tensor): Index tensor (N, *).

        Returns:
            torch.Tensor: Multiclass Sorensen-Dice loss.
        """
        inputs = torch.atleast_3d(F.softmax(inputs, 1)).flatten(2)
        targets = torch.atleast_3d(torch.movedim(F.one_hot(targets, inputs.shape[1]), -1, 1)).flatten(2)
        return 1 - ((2 * (inputs * targets).sum(2) + self.epsilon) / (inputs.sum(2) + targets.sum(2) + self.epsilon)).mean()
