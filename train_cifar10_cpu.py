# -*- coding: utf-8 -*-
'''

Train CIFAR10 script enabled for CPU! Runs an adjustable smaller subset with CUDA disabled.

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer
from torch.utils.data import Subset

# debugging
import ipdb
# use ipdb.set_trace() to put breakpoints, n for next, c for close (loop), q for quit

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--noaug', action='store_false', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='vit')
parser.add_argument('--dp', action='store_true', help='use data parallel')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--subsize', type=int, default='100')

args = parser.parse_args()

# take in args
usewandb = not args.nowandb

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp
aug = args.noaug

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))


# Prepare dataset
subset_size = args.subsize

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
training_indices = np.random.choice(len(trainset), subset_size, replace=False)
training_subset = Subset(trainset, training_indices)

trainloader = torch.utils.data.DataLoader(training_subset, batch_size=bs, shuffle=True, num_workers=1)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testing_indices = np.random.choice(len(testset), subset_size, replace=False)
testing_subset = Subset(testset, testing_indices)

testloader = torch.utils.data.DataLoader(testing_subset, batch_size=100, shuffle=False, num_workers=1)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')

# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = 10
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=10,
                downscaling_factors=(2,2,2,1))

# For Multi-GPU
if 'cuda' in device:
    print(device)
    if args.dp:
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/{}-ckpt.t7'.format(args.net))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def squentropy(outputs, targets):
    num_classes = len(classes)
    print(f'\ntargets.size: {torch.zeros([targets.size()[0], num_classes])}\n')
    print(device)

    target_final = torch.zeros([targets.size()[0], num_classes], device=device).scatter_(1, targets.reshape(
        targets.size()[0], 1), 1)

    # ce_func = nn.CrossEntropyLoss().cuda()
    ce_func = nn.CrossEntropyLoss()
    print(f'outputs: {outputs[target_final == 1]}')
    loss = (torch.sum(outputs ** 2) - torch.sum((outputs[target_final == 1]) ** 2)) / (
                num_classes - 1) / target_final.size()[0] \
            + ce_func(outputs, targets)
    return loss

# Loss function is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

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

##### Training
# scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    num_traindata = len(trainloader)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            loss = squentropy(outputs, targets)
        # scaler.scale(loss).backward()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.step(optimizer)
        # scaler.update()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, num_traindata, 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (train_loss/(num_traindata), 100.*correct/total, correct, total))
    return train_loss/(num_traindata)

##### Validation
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    num_testdata = len(testloader)

    logits_list = []
    labels_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = squentropy(outputs, targets)

            logits_list.append(outputs)
            labels_list.append(targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, num_testdata, 'Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
                % (test_loss/num_testdata, 100.*correct/total, correct, total))
    
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {"model": net.state_dict(),
              "optimizer": optimizer.state_dict(),
            #   "scaler": scaler.state_dict()
              }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    # Calculate ECE
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    ece = _ECELoss().forward(logits, labels).item()
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {(test_loss/num_testdata):.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss/(num_testdata), acc, ece

def main():
    if usewandb:
        import wandb
        watermark = "{}_lr{}".format(args.net, args.lr)
        wandb.init(project="cifar10-challenge",
                name=watermark + " (DATASET=10K)")
        wandb.config.update(args)

    list_loss = []
    list_acc = []

    if usewandb:
        wandb.watch(net)
        
    global_start_time = time.time()
    # uncomment this after gpu access
    # net.cuda()
    for epoch in range(start_epoch, args.n_epochs):
        print(f"Epoch: {epoch}/{args.n_epochs}")
        start = time.time()
        trainloss = train(epoch)
        val_loss, acc, ece = test(epoch)
        
        scheduler.step(epoch-1) # step cosine scheduling
        
        list_loss.append(val_loss)
        list_acc.append(acc)
        
        # Log training..
        if usewandb:
            wandb.log({'Train Loss': trainloss}, step=epoch)
            wandb.log({'Eval Loss': val_loss}, step=epoch)
            wandb.log({'Eval Acc': acc}, step=epoch)
            wandb.log({'Learning Rate': optimizer.param_groups[0]["lr"]}, step=epoch)
            wandb.log({'Testing ECE': ece}, step=epoch)
            wandb.log({"Epoch Runtime (s)": time.time() - start}, step=epoch)
            wandb.log({"Total Runtime (s)": time.time() - global_start_time}, step=epoch)
            wandb.log({'Epoch': epoch})

        # Write out csv..
        with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss) 
            writer.writerow(list_acc) 
        print(list_loss)

    # writeout wandb
    if usewandb:
        wandb.save("wandb_{}.h5".format(args.net))

# running on cpu doesn't allow multiprocessing
if __name__ == '__main__':
    main()