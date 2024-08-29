# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.backends.cudnn as cudnn
# import numpy as np

# import torchvision
# import torchvision.transforms as transforms

# from randomaug import RandAugment

# '''
# Custom torchvision transformer that reshapes data tensors into the given vector input; 
# Given size must make sense-- cannot reshape (100,2,5,5) tensor into (5,5,5) tensor

# See https://pytorch.org/docs/stable/generated/torch.Tensor.view.html for more info
# '''
# class ReshapeTransform:
#     def __init__(self, new_shape):
#         self.new_shape = new_shape

#     def __call__(self, x):
#         return x.view(*self.new_shape)

# '''
# Dataloader class
# '''
# class Dataloader:
#     # maybe keep, maybe delete
#     def load_transforms(dataset_arg, aug, image_size):
#         transform_train = None
#         transform_test = None
#         if (dataset_arg == "cifar10") or (dataset_arg == "cifar100") or (dataset_arg == "svhn"):
#             transform_train = transforms.Compose([
#                 transforms.RandomCrop(32, padding=4),
#                 transforms.Resize(image_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ])
#             transform_test = transforms.Compose([
#                 transforms.RandomCrop(32, padding=4),
#                 transforms.Resize(image_size),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#             ])
#         elif dataset_arg == "mnist":
#             transform_train = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,)),
#                 ReshapeTransform((1, 784))
#             ])
#             transform_test = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize((0.1307,), (0.3081,)),
#                 ReshapeTransform((1, 784))
#             ])
        
#         # Add RandAugment with N, M(hyperparameter)
#         if args.aug:  
#             N = 2; M = 14;
#             transform_train.transforms.insert(0, RandAugment(N, M))

#         return transform_train, transform_test
    
#     def load_train_test_sets(dataset_arg, augment_transforms, image_size):
#         trainset = None
#         testset = None

#         transform_train, transform_test = Dataloader.load_transforms(dataset_arg, augment_transforms, image_size)

#         if dataset_arg == "cifar10":
#             trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, 
#                                                     transform=transform_train)
#             testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, 
#                                                     transform=transform_test)
#         elif dataset_arg == "mnist":
#             trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
#                                                     transform=transform_train)
#             testset = torchvision.datasets.MNIST(root='./data', train=False, download=True,
#                                                     transform=transform_test)
#         elif dataset_arg == "svhn":
#             trainset = torchvision.datasets.SVHN(root='./data', split='train', download=True, 
#                                                  transform=transform_train)
#             testset = torchvision.datasets.SVHN(root='./data', split='test', download=True, 
#                                                 transform=transform_test)
#         elif dataset_arg == "cifar100":
#             trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, 
#                                                     transform=transform_train)
#             testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, 
#                                                     transform=transform_test)
#         else:
#             raise Exception(f'\nInvalid dataset function input: {args.dataset} \
#                                         \nPlease input a valid dataset as input to the dataset parameter\n')
#         return trainset, testset
    
#     # Call this function to load final training and testing loaders
#     def load_train_test_loaders(trainset, testset, batch_sz):
#         trainset, testset = Dataloader.load_train_test_sets(dataset_arg, imsize)

#         trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_sz, shuffle=True, num_workers=8)
#         testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

#         return trainloader, testloader
