import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import ipdb

class Learnable_Squentropy(nn.Module):
    # rescale factor of 1 yields standard non-rescaled squentropy
    # if True, negative classification sets incorrect classes in one-hot-encoding to -1
    def __init__(self, device, num_classes=10, rescale_factor=1, neg_class=False):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.rescale_factor = rescale_factor
        self.negative_classification = neg_class
    
    # rescale factor will be retrieved from network
    def forward(self, outputs, targets, learnable_rescale_factor=None):
        num_classes = self.num_classes

        # one-hot encoding of target
        target_final = torch.zeros([targets.size()[0], num_classes], device = self.device).scatter_(1, targets.reshape(
            targets.size()[0], 1), 1)
        
        rescale_factor = self.rescale_factor
        if learnable_rescale_factor is not None:
            rescale_factor = learnable_rescale_factor

        # ce_func = nn.CrossEntropyLoss().cuda()
        ce_func = nn.CrossEntropyLoss()
        loss = rescale_factor * (torch.sum(outputs ** 2) - torch.sum((outputs[target_final == 1]) ** 2)) / (
                    num_classes - 1) / target_final.size()[0] \
                + ce_func(outputs, targets)
        return loss

class Squentropy(nn.Module):
    # rescale factor of 1 yields standard non-rescaled squentropy
    # if True, negative classification sets incorrect classes in one-hot-encoding to -1
    def __init__(self, device, num_classes=10, rescale_factor=1, neg_class=False):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.rescale_factor = rescale_factor
        self.negative_classification = neg_class
    
    def forward(self, outputs, targets):
        num_classes = self.num_classes

        # create one-hot encoding of target
        if self.negative_classification:
            target_final = torch.zeros([targets.size()[-1], num_classes], device=self.device).scatter_(1, targets.reshape(
                targets.size()[0], 1), 1)
        else: 
            target_final = torch.zeros([targets.size()[0], num_classes], device=self.device).scatter_(1, targets.reshape(
                targets.size()[0], 1), 1)

        # ce_func = nn.CrossEntropyLoss().cuda()
        ce_func = nn.CrossEntropyLoss()
        loss = self.rescale_factor * (torch.sum(outputs ** 2) - torch.sum((outputs[target_final == 1]) ** 2)) / (
                    num_classes - 1) / target_final.size()[0] \
                + ce_func(outputs, targets)
        return loss

class Rescaled_MSE(nn.Module):
    def __init__(self, device, num_classes=10, sqLoss_t=1, sqLoss_M=1):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        self.sqLoss_t = sqLoss_t
        self.sqLoss_M = sqLoss_M
    
    def forward(self, outputs, targets):
        num_classes = self.num_classes

        # one-hot encoding of target
        target_final = torch.zeros([targets.size()[0], num_classes], device=self.device).scatter_(1, targets.reshape(
            targets.size()[0], 1), 1)
        mse_weights = target_final * self.sqLoss_t + 1
        
        loss = torch.mean((outputs - self.sqLoss_M * target_final.type(torch.float)) ** 2 * mse_weights)
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

# SCE = Static Calibration Error; should be better model for calibration than ECE for multiclassification
# SCE model from https://openaccess.thecvf.com/content_CVPRW_2019/papers/Uncertainty%20and%20Robustness%20in%20Deep%20Visual%20Learning/Nixon_Measuring_Calibration_in_Deep_Learning_CVPRW_2019_paper.pdf
class _SCELoss(nn.Module):
    def __init__(self, n_classes, n_bins=10):
        super(_SCELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.n_classes = n_classes
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        num_predictions = len(logits)
        sce = torch.zeros(1, device=logits.device)

        class_confidences = [[] for n in range(self.n_classes)]
        class_labels = [[] for n in range(self.n_classes)]

        index = 0
        for confidence, prediction in zip(confidences, predictions):
            class_confidences[prediction].append(confidence.item())
            class_labels[prediction].append(labels[index].item())
            index += 1
        
        for current_class_num, class_confidence in enumerate(class_confidences):
            for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
                confidences = torch.tensor(class_confidence)
                in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
                num_in_bin = in_bin.int().sum()
                prop_in_bin = num_in_bin / num_predictions

                if prop_in_bin.item() > 0:
                    predictions = torch.ones(int(num_in_bin.item())) * current_class_num
                    accuracy_in_bin = predictions.eq(torch.tensor(class_labels[current_class_num])).float().mean()
                    avg_confidence_in_bin = confidences.mean()
                    sce += torch.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin

        sce = sce / self.n_classes
        print('sce = ', sce.item())
        return sce