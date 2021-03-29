from torch.nn import Module
import torch
import numpy as np


def categorical_accuracy(y_pred, y_true):
    max_vals, max_indices = torch.max(y_pred, 1)
    train_acc = (max_indices == y_true).data.to('cpu').numpy().mean()
    return train_acc


class CategoricalAccuracy(Module):

    def __init__(self):
        super(CategoricalAccuracy, self).__init__()
        self.__name__ = 'Accuracy'
        self.pattern = '{:.3f}'

    def forward(self, y_pred, y_true):
        return categorical_accuracy(y_pred, y_true)


def top_k_accuracy(y_pred, y_true, top_k=1):
    dist, ranks = torch.topk(y_pred, top_k)
    y_comp = torch.transpose(y_true.repeat(top_k).view(top_k, -1), 0, 1)
    train_acc = (ranks == y_comp).float().sum(axis=1).mean()
    return train_acc


class TopKAccuracy(Module):

    def __init__(self, top_k=1):
        super(TopKAccuracy, self).__init__()
        self.__name__ = f'Top-{top_k} Accu.'
        self.pattern = '{:.3f}'
        self.top_k = top_k

    def forward(self, y_pred, y_true):
        return top_k_accuracy(y_pred, y_true, self.top_k)


class CrossEntropyLoss(Module):

    def __init__(self, one_hot_encoding=False, *args, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        self.__name__ = f'CrossEntropyLoss'
        self.pattern = '{:.4f}'
        self.one_hot_encoding = one_hot_encoding
        self.func = torch.nn.CrossEntropyLoss(*args, **kwargs)

    def forward(self, y_pred, y_true):
        if self.one_hot_encoding:
            _, y_true = y_true.max(dim=-1)
        return self.func(y_pred, y_true)


def rmse(y_pred, y_true):
    return torch.sqrt(torch.mean((y_pred-y_true)**2))


class RMSELoss(Module):
    def __init__(self, eps=1e-6, round=False, from_root=False):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        self.round = round
        self.from_root = from_root
        self.__name__ = 'RMSE'
        self.pattern = '{:.3f}'

    def forward(self, y_pred, y_true):
        if self.from_root:
            y_pred = y_pred**2
        if self.round:
            y_pred = torch.round(y_pred)
        return torch.sqrt(self.mse(y_pred, y_true) + self.eps)


class wRMSELoss(Module):
    def __init__(self, w=0.75, eps=1e-6, round=False, from_root=False):
        super().__init__()
        self.w = w
        self.eps = eps
        self.round = round
        self.from_root = from_root
        self.__name__ = 'wRMSE'
        self.pattern = '{:.3f}'

    def forward(self, y_pred, y_true):
        if self.from_root:
            y_pred = y_pred**2
        if self.round:
            y_pred = torch.round(y_pred)
        if len(y_true.size()) >= 2:
            w = torch.from_numpy(np.array([self.w ** s1 for s1 in range(y_true.size(1))], np.float32)).to(y_true.device)
            return torch.sqrt(torch.sum(w * (y_pred-y_true)**2)/w.sum()/y_pred.size(0) + self.eps)
        else:
            return torch.sqrt(torch.mean((y_pred - y_true) ** 2) + self.eps)


class MSRELoss(Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.__name__ = 'MSRE'
        self.pattern = '{:.3f}'

    def forward(self, y_pred, y_true):
        y_true = torch.sqrt(y_true)
        y_pred = torch.sqrt(torch.sign(y_pred)*y_pred)*torch.sign(y_pred)

        return self.mse(y_pred, y_true)


class MSELoss(Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.__name__ = 'MSE'
        self.pattern = '{:.3f}'

    def forward(self, y_pred, y_true):
        return self.mse(y_pred, y_true)


class MAELoss(Module):
    def __init__(self):
        super().__init__()
        self.mae = torch.nn.L1Loss()
        self.__name__ = 'MAE'
        self.pattern = '{:.3f}'

    def forward(self, y_pred, y_true):
        return self.mae(y_pred, y_true)


class SmoothL1Loss(Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta
        self.sl1 = torch.nn.SmoothL1Loss(self.beta)
        self.__name__ = f'SmoothL1[b={self.beta}]'
        self.pattern = '{:.3f}'

    def forward(self, y_pred, y_true):
        return self.sl1(y_pred, y_true)