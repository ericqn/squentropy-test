import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

class Loss_Functions:
    def __init__(self, device, num_classes, dataset):
        self.device = device
        self.num_classes = num_classes
        # Standard MSE rescale variables
        self.sqLoss_M = 1
        self.sqLoss_t = 1
        # For Rescaled Squentropy variables
        self.sqen_alpha = 1

        if (dataset == 'cifar10') or (dataset == 'ciffar100'):
            self.sqLoss_M = 10

    '''
    Squentropy function
    '''
    def squentropy(self, outputs, targets):
        num_classes = self.num_classes
        used_device = self.device

        # one-hot encoding of target
        target_final = torch.zeros([targets.size()[0], num_classes], device=used_device).scatter_(1, targets.reshape(
            targets.size()[0], 1), 1)

        # ce_func = nn.CrossEntropyLoss().cuda()
        ce_func = nn.CrossEntropyLoss()
        loss = (torch.sum(outputs ** 2) - torch.sum((outputs[target_final == 1]) ** 2)) / (
                    num_classes - 1) / target_final.size()[0] \
                + ce_func(outputs, targets)
        return loss

    # Rescaled Mean Square Error function
    def rescaled_mse(self, outputs, targets):
        num_classes = self.num_classes

        # one-hot encoding of target
        target_final = torch.zeros([targets.size()[0], num_classes], device=device).scatter_(1, targets.reshape(
            targets.size()[0], 1), 1)
        mse_weights = target_final * sqLoss_t + 1
        
        loss = torch.mean((outputs - sqLoss_M * target_final.type(torch.float)) ** 2 * mse_weights)
        return loss

    # Default Cross Entropy given by Pytorch
    def cross_entropy(self, outputs, targets):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, targets)
        return loss

# Calculating ECE
class _ECELoss(nn.Module):
    def __init__(self, n_bins=10):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                # print('bin_lower=%f, bin_upper=%f, accuracy=%.4f, confidence=%.4f: ' % (bin_lower, bin_upper, accuracy_in_bin.item(),
                #       avg_confidence_in_bin.item()))
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        print('ece = ', ece)
        return ece