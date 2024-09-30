# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
Original codebase written by @kentaroy47, @arutema47


'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import Subset
import numpy as np

import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from models.vit import ViT
from models.convmixer import ConvMixer

from utils import progress_bar
from randomaug import RandAugment
from train_functions import _ECELoss, Loss_Functions

import ipdb

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
parser.add_argument('--bs', type=int, default='512')
parser.add_argument('--size', type=int, default="32")
parser.add_argument('--n_epochs', type=int, default='200')
parser.add_argument('--n_classes', type=int, default='10')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")
parser.add_argument('--loss_eq', default='sqen')
parser.add_argument('--dataset', default='cifar10')
parser.add_argument('--subset_prop', default='-1', type=float, help='sets the proportion of the training subset to be used')
parser.add_argument('--sqen_alpha', default='-1', type=float, help='set rescale value for squentropy loss function')

args = parser.parse_args()

# take in args
usewandb = not (args.nowandb)
use_sqen_rs = False
if (args.sqen_alpha > -1) and ((args.loss_eq == 'sqen_rs') or (args.loss_eq == 'sqen_neg_rs')):
    use_sqen_rs = True
if usewandb:
    import wandb
    watermark = "{}_model:{}_loss:{}_lr:{}".format(args.dataset, args.net, args.loss_eq, args.lr)
    if (use_sqen_rs):
        watermark = f"{args.dataset}_model:{args.net}_loss:{args.loss_eq}_alpha:{args.sqen_alpha}_lr:{}"
    wandb.init(project="Squentropy Testing 3 - rescalable",
            name=watermark)
    wandb.config.update(args)

use_amp = not args.noamp
aug = args.noaug

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

if args.net=="vit_timm":
    args.size = 384

# Rescaling Variables
# sqLoss_t = 1
# sqLoss_M = 1
# sqen_alpha = 1

# if (args.dataset == 'cifar10') or (args.dataset == 'cifar100'):
#     sqLoss_M = 10

'''
Custom torchvision transformer that reshapes data tensors into the given vector input; 
Given size must make sense-- cannot reshape (100,2,5,5) tensor into (5,5,5) tensor

See https://pytorch.org/docs/stable/generated/torch.Tensor.view.html for more info
'''
class ReshapeTransform:
    def __init__(self, new_shape):
        self.new_shape = new_shape

    def __call__(self, x):
        return x.view(*self.new_shape)

'''
Dataloader class
'''
class Dataloader:

    # Load data transformers based on dataset argument. Can also toggle whether or not data
    # should be augmented or not
    def load_transforms(dataset_arg):
        transform_train = None
        transform_test = None

        image_sz = args.size

        if (dataset_arg == "cifar10") or (dataset_arg == "cifar100"):
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.Resize(image_sz),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
            transform_test = transforms.Compose([
                transforms.Resize(image_sz),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])
        elif (dataset_arg == "svhn"):
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
        elif dataset_arg == "mnist":
            transform_train = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ReshapeTransform((1, 784))
            ])
            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
                ReshapeTransform((1, 784))
            ])

            # args.bs = 64
        else:
            raise Exception(f'\nInvalid dataset function input: {dataset_arg} \
                                        \nPlease input a valid dataset as input to the dataset parameter\n')
        
        if (transform_train == None) or (transform_test == None):
            raise Exception(f'\nInvalid dataset function input: {dataset_arg} \
                                        \nPlease input a valid dataset as input to the dataset parameter\n')
        
        # Add RandAugment with N, M(hyperparameter)
        if aug:  
            N = 2; M = 14;
            
            if dataset_arg == "mnist":
                grayscale = True
            else:
                grayscale = False
            
            transform_train.transforms.insert(0, RandAugment(N, M, grayscale))

        return transform_train, transform_test
    
    # Note: Modifies global variables args.n_classes
    def load_train_test_sets(dataset_arg):
        trainset = None
        testset = None

        transform_train, transform_test = Dataloader.load_transforms(dataset_arg)

        if dataset_arg == "cifar10":
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, 
                                                    transform=transform_train)
            testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, 
                                                    transform=transform_test)
        elif dataset_arg == "mnist":
            
            trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                                    transform=transform_train)
            testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
                                                    transform=transform_test)
        elif dataset_arg == "svhn":
            trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, 
                                                 transform=transform_train)
            testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, 
                                                transform=transform_test)
        elif dataset_arg == "cifar100":
            trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, 
                                                    transform=transform_train)
            testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, 
                                                    transform=transform_test)
            args.n_classes = 100
        else:
            raise Exception(f'\nInvalid dataset function input: {dataset_arg} \
                                        \nPlease input a valid dataset as input to the dataset parameter\n')
        
        if (args.subset_prop > -1):
            train_subset_size = int(args.subset_prop * len(trainset))
            test_subset_size = int(args.subset_prop * len(testset))

            if (train_subset_size < 30) or (test_subset_size < 10):
                raise Exception(f'\nSubset proportion of {args.subset_prop} is too small.')

            training_indices = np.random.choice(len(trainset), train_subset_size, replace=False)
            trainset = Subset(trainset, training_indices)
            testing_indices = np.random.choice(len(testset), test_subset_size, replace=False)
            testset = Subset(testset, testing_indices)

        return trainset, testset
    
    # Call this function to load final training and testing loaders given train and testing sets
    def load_train_test_loaders(trainset, testset):
        batch_sz = args.bs
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, shuffle=True, num_workers=8)
        testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

        return trainloader, testloader

'''
Network Factory class
'''
class Network_Factory:
    def load_model(model_name):
        # Model factory..
        print('==> Building model..')

        network_model = None
        image_sz = args.size
        n_classes = args.n_classes

        if (model_name == 'res18'):
            network_model = ResNet18()
        elif (model_name == 'vgg'):
            # net used in squentropy paper
            network_model = torchvision.models.vgg11_bn(weights=None, num_classes=n_classes)
            # net = VGG('VGG19')
        elif (model_name=='res34'):
            network_model = ResNet34()
        elif (model_name == 'res50'):
            network_model = ResNet50()
        elif (model_name == 'res101'):
            network_model = ResNet101()
        elif (model_name == 'wide_res'):
            from models.wide_resnet import WideResNet
            network_model = WideResNet(num_classes=n_classes)
        elif (model_name =='tcnn'):
            from models.tcnn import TCN
            # Default hyperparam config from repo
            hidden_layer_units = 25
            num_levels = 8
            
            network_model = TCN(
                input_size=1, 
                output_size=n_classes, 
                num_channels=[hidden_layer_units]*num_levels,
                kernel_size=7,
                dropout=0.05
            )
        elif (model_name =="convmixer"):
            # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
            network_model = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=n_classes)
        elif (model_name =="mlpmixer"):
            from models.mlpmixer import MLPMixer
            network_model = MLPMixer(
                image_size = image_sz,
                channels = 3,
                patch_size = args.patch,
                dim = 512,
                depth = 6,
                num_classes = n_classes
            )
        elif (model_name == "vit_small"):
            from models.vit_small import ViT
            network_model = ViT(
                image_size = image_sz,
                patch_size = args.patch,
                num_classes = n_classes,
                dim = int(args.dimhead),
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        elif (model_name =="vit_tiny"):
            from models.vit_small import ViT
            network_model = ViT(
                image_size = image_sz,
                patch_size = args.patch,
                num_classes = n_classes,
                dim = int(args.dimhead),
                depth = 4,
                heads = 6,
                mlp_dim = 256,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        elif (model_name == "simplevit"):
            from models.simplevit import SimpleViT
            network_model = SimpleViT(
                image_size = image_sz,
                patch_size = args.patch,
                num_classes = n_classes,
                dim = int(args.dimhead),
                depth = 6,
                heads = 8,
                mlp_dim = 512
            )
        elif (model_name == "vit"):
            from models.vit import ViT
            # ViT initialized for cifar10
            network_model = ViT(
                image_size = image_sz,
                patch_size = args.patch,
                num_classes = n_classes,
                dim = int(args.dimhead),
                depth = 6,
                heads = 8,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1
            )
        elif (model_name == "vit_timm"):
            import timm
            network_model = timm.create_model("vit_base_patch16_384", pretrained=True)
            network_model.head = nn.Linear(net.head.in_features, 10)
        elif (model_name == "cait"):
            from models.cait import CaiT
            network_model = CaiT(
                image_size = image_sz,
                patch_size = args.patch,
                num_classes = n_classes,
                dim = int(args.dimhead),
                depth = 6,   # depth of transformer for patch to patch attention only
                cls_depth=2, # depth of cross attention of CLS tokens to patch
                heads = 8,
                mlp_dim = 512,
                dropout = 0.1,
                emb_dropout = 0.1,
                layer_dropout = 0.05
            )
        elif (model_name == "cait_small"):
            from models.cait import CaiT
            network_model = CaiT(
                image_size = image_sz,
                patch_size = args.patch,
                num_classes = n_classes,
                dim = int(args.dimhead),
                depth = 6,   # depth of transformer for patch to patch attention only
                cls_depth=2, # depth of cross attention of CLS tokens to patch
                heads = 6,
                mlp_dim = 256,
                dropout = 0.1,
                emb_dropout = 0.1,
                layer_dropout = 0.05
            )
        elif (model_name == "swin"):
            from models.swin import swin_t
            network_model = swin_t(window_size=args.patch,
                        num_classes=n_classes,
                        downscaling_factors=(2,2,2,1))
        
        return network_model

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainset, testset = Dataloader.load_train_test_sets(args.dataset)
trainloader, testloader = Dataloader.load_train_test_loaders(trainset, testset)

model_name = args.net
net = Network_Factory.load_model(model_name)

# For viewing data (debugging purposes):

# train_data_iter = iter(trainloader)
# test_data_iter = iter(testloader)
# train_image, train_label = next(train_data_iter)
# test_image, test_label = next(test_data_iter)
# ipdb.set_trace()

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

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

##### Training
def train(epoch):
    net.train()

    n_classes = args.n_classes
    dataset = args.dataset
    loss_func = Loss_Functions(device=device, num_classes=n_classes, dataset=dataset)

    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Train with amp
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = net(inputs)
            if (args.loss_eq == 'sqen'):
                alpha = args.sqen_alpha
                loss = loss_func.squentropy(outputs, targets)
            elif (args.loss_eq == 'cross'):
                loss = loss_func.cross_entropy(outputs, targets)
            elif (args.loss_eq == 'mse'):
                loss = loss_func.rescaled_mse(outputs, targets)
            elif(args.loss_eq == 'sqen_rs'):
                alpha = args.sqen_alpha
                loss = loss_func.rescaled_squentropy(outputs, targets, alpha)
            elif(args.loss_eq == 'sqen_neg_rs'):
                alpha = args.sqen_alpha
                loss = loss_func.rescaled_negative_squentropy(outputs, targets, alpha)
            else:
                raise Exception(f'\nInvalid loss function input: {args.loss_eq} \
                                \nPlease input \'sqen\', \'cross\', or \'mse\' as inputs to the loss_eq parameter\n')
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss/(batch_idx+1)

##### Validation
def test(epoch):
    net.eval()

    n_classes = args.n_classes
    dataset = args.dataset
    loss_func = Loss_Functions(device=device, num_classes=n_classes, dataset=dataset)
    ece_func = _ECELoss()

    global best_acc
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

            # determining loss function
            if (args.loss_eq == 'sqen'):
                loss = loss_func.squentropy(outputs, targets)
            elif (args.loss_eq == 'cross'):
                loss = loss_func.cross_entropy(outputs, targets)
            elif (args.loss_eq == 'mse'):
                loss = loss_func.rescaled_mse(outputs, targets)
            elif(args.loss_eq == 'sqen_rs'):
                alpha = 0.1
                loss = loss_func.rescaled_squentropy(outputs, targets, alpha)
            elif(args.loss_eq == 'sqen_neg_rs'):
                alpha = 1
                loss = loss_func.rescaled_negative_squentropy(outputs, targets, alpha)
            else:
                raise Exception(f'\nInvalid loss function input: {args.loss_eq} \
                                \nPlease input \'sqen\', \'cross\', or \'mse\' as inputs to the loss_eq parameter\n')

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
              "scaler": scaler.state_dict()
              }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
        best_acc = acc
    
    # Calculate ECE
    logits = torch.cat(logits_list)
    labels = torch.cat(labels_list)
    ece = ece_func.forward(logits, labels).item()
    
    os.makedirs("log", exist_ok=True)
    content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {(test_loss/num_testdata):.5f}, acc: {(acc):.5f}'
    print(content)
    with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
        appender.write(content + "\n")
    return test_loss/(num_testdata), acc, ece

if usewandb:
    wandb.watch(net)
    
net.cuda()
global_start_time = time.time()

# tracking ECE min, max, avg
list_loss = []
list_acc = []

ece_sum = 0
ece_max = 0
ece_min = 100
eval_acc_max = 0

for epoch in range(start_epoch, args.n_epochs):
    print(f"Epoch: {epoch}/{args.n_epochs}")
    start = time.time()
    trainloss = train(epoch)
    val_loss, acc, ece = test(epoch)

    ece_sum += ece
    ece_avg = ece_sum / (epoch + 1)
    if ece > ece_max:
        ece_max = ece

    if ece < ece_min:
        ece_min = ece

    if acc > eval_acc_max:
        eval_acc_max = acc
    
    # scheduler.step(epoch-1) # step cosine scheduling
    scheduler.step() # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # TODO: Change step into epoch
    # Log training..
    if usewandb:
        wandb.log({'Best Eval Acc': eval_acc_max}, step=epoch)
        wandb.log({'Max ECE': ece_max}, step=epoch)
        wandb.log({'Min ECE': ece_min}, step=epoch)
        wandb.log({'Avg ECE': ece_avg}, step=epoch)
        wandb.log({'Train Loss': trainloss}, step=epoch)
        wandb.log({'Eval Loss': val_loss}, step=epoch)
        wandb.log({'Eval Acc': acc}, step=epoch)
        wandb.log({'Learning Rate': optimizer.param_groups[0]["lr"]}, step=epoch)
        wandb.log({'Testing ECE': ece}, step=epoch)
        wandb.log({"Epoch Runtime (s)": time.time() - start}, step=epoch)
        wandb.log({"Total Runtime (s)": time.time() - global_start_time}, step=epoch)
        wandb.log({'Epoch': epoch})

        # log max, min, avg for ECE and test loss

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    print(list_loss)

# writeout wandb
if usewandb:
    wandb.save("wandb_{}.h5".format(args.net))