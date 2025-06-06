import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

# Example ECE implementation from your code
class _ECELoss(nn.Module):
    def __init__(self, n_bins=10):
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        # softmaxes = F.softmax(logits, dim=1)
        softmaxes = logits
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)
        ece = torch.zeros(1, device=logits.device)
        ipdb.set_trace()

        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        print('ece = ', ece.item())
        return ece

# Create toy logits and labels
logits = torch.tensor([[0.78, 0.22], [0.36, 0.64], 
                       [0.08, 0.92], [0.58, 0.42], 
                       [0.49, 0.51], [0.85, 0.15], 
                       [0.30, 0.70], [0.63, 0.37],
                       [0.17, 0.83]])
labels = torch.tensor([0, 1, 0, 0, 0, 0, 1, 1, 1])

# Initialize the ECE class
ece_criterion = _ECELoss(n_bins=5)

# Calculate ECE
ece_value = ece_criterion(logits, labels)
print('Calculated ECE:', ece_value.item())