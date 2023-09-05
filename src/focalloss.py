import torch
from torch import nn
import torch.nn.functional as F

class SegmentationFocalLoss(nn.Module):
    def __init__(self, gamma=2, weights: list[float] = None, logits=True):
        """
        :param gamma: 簡単なサンプルの重み. 大きいほど簡単なサンプルを重視しない.
        :param weights: weights by classes,
        :param logits:
        """
        super().__init__()
        self.gamma = gamma
        self.class_weight_tensor = try_cuda(torch.tensor(weights).view(-1, 1, 1)) if weights else None
        self.logits = logits
        if not logits and weights is not None:
            RuntimeWarning("重みを適用するにはlogitsをTrueにしてください.")

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor) -> float:
        '''
        :param pred: batch_size, n_classes, height, width
        :param teacher: batch_size, n_classes, height, width
        :return:
        '''
        if self.logits:
            ce_loss = F.binary_cross_entropy_with_logits(pred, teacher, reduce=False)
            pt = torch.exp(-ce_loss)
            if self.class_weight_tensor:
                class_weight_tensor = self.class_weight_tensor.expand(pred.shape[0],
                                                                      self.class_weight_tensor.shape[0],
                                                                      self.class_weight_tensor.shape[1],
                                                                      self.class_weight_tensor.shape[2])
                focal_loss = (1. - pt) ** self.gamma * (ce_loss * class_weight_tensor)
            else:
                focal_loss = (1. - pt) ** self.gamma * ce_loss
        else:
            ce_loss = F.cross_entropy(pred, teacher.argmax(1), reduce=False)
            pt = torch.exp(-ce_loss)
            focal_loss = (1. - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)
    

class ClassificationFocalLoss(nn.Module):
    def __init__(self, n_classes: int, gamma=2, weights: list[float] = None, logits=True):
        """
        :param n_classes: 
        :param gamma: 簡単なサンプルの重み. 大きいほど簡単なサンプルを重視しない.
        :param weights: weights by classes,
        :param logits:
        """
        super().__init__()
        self._n_classes = n_classes
        self.gamma = gamma
        self.class_weight_tensor = try_cuda(torch.tensor(weights).view(-1, )) if weights else None
        self.logits = logits
        if not logits and weights is not None:
            RuntimeWarning("重みを適用するにはlogitsをTrueにしてください.")

    def forward(self, pred: torch.Tensor, teacher: torch.Tensor) -> float:
        """
        :param pred: batch_size, n_classes
        :param teacher: batch_size, 
        :return: 
        """
        if self.logits:
            if teacher.ndim == 1:  # 1次元ならonehotの2次元tensorにする
                teacher = torch.eye(self._n_classes)[teacher]
            ce_loss = F.binary_cross_entropy_with_logits(pred, teacher, reduce=False)
            pt = torch.exp(-ce_loss)

            if self.class_weight_tensor:
                class_weight_tensor = self.class_weight_tensor.expand(pred.shape[0],
                                                                      self.class_weight_tensor.shape[0], )
                focal_loss = (1. - pt) ** self.gamma * (ce_loss * class_weight_tensor)
            else:
                focal_loss = (1. - pt) ** self.gamma * ce_loss
        else:
            if teacher.ndim == 2:  # onehotの2次元tensorなら1次元にする
                teacher = teacher.argmax(1)
            ce_loss = F.cross_entropy(pred, teacher, reduce=False)
            pt = torch.exp(-ce_loss)
            focal_loss = (1. - pt) ** self.gamma * ce_loss
        return torch.mean(focal_loss)