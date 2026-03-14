import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import my_data_class

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import proposed_model
from utils import progress_bar
from models.transformer_model import HybridResNetTransformer
from models.arcface import ArcMarginProduct
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
import numpy as np
import random

def mixup_data(x, y, alpha=0.2, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def augment_tensor(x, std=0.05):
    """Apply augmentations directly on GPU tensors.
    Includes: Gaussian noise, random horizontal flip, brightness/contrast jitter, random blur.
    """
    # 1. Gaussian noise
    x = x + torch.randn_like(x) * std
    
    # 2. Random horizontal flip (50% chance)
    if random.random() > 0.5:
        x = torch.flip(x, dims=[3])
    
    # 3. Brightness jitter: scale by random factor in [0.8, 1.2]
    brightness_factor = 0.8 + random.random() * 0.4
    x = x * brightness_factor
    
    # 4. Contrast jitter: blend toward mean
    if random.random() > 0.5:
        contrast_factor = 0.8 + random.random() * 0.4
        mean_val = x.mean()
        x = contrast_factor * x + (1 - contrast_factor) * mean_val
    
    return x

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs):
    """Linear warmup for `warmup_epochs`, then cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

import math
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(
        description="Split dataset into train/test for face recognition."
    )
    parser.add_argument('--train_data', default="/storage4tb/PycharmProjects/Datasets/lensless_data/train/ymdct_npy", type=str, required=False, help='Path to train data')
    parser.add_argument('--test_data', default="/storage4tb/PycharmProjects/Datasets/lensless_data/test/ymdct_npy", type=str, required=False, help='Path to test/validation data')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--lr', default=0.05, type=float, help='Learning rate')
    parser.add_argument('--num_workers', default=3, type=int, help='Number of workers')
    parser.add_argument('--num_epoch', default=120, type=int, help='Number of epochs')
    parser.add_argument('--model', default='cnn', type=str, choices=['cnn', 'transformer'], help='Model type: cnn or transformer')
    parser.add_argument('--warmup_epochs', default=5, type=int, help='Number of warmup epochs')
    parser.add_argument('--freeze_epochs', default=10, type=int, help='Freeze early ResNet layers for this many epochs')
    parser.add_argument('--resume', default=None, type=str, help='Path to last.pth checkpoint to resume training from')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_cuda = torch.cuda.is_available()
    best_acc = 0
    start_epoch = 0

    print('==> Preparing data..')
    trainset = my_data_class.Lensless_DCT_offline(args.train_data)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    testset = my_data_class.Lensless_DCT_offline(args.test_data)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    now = datetime.now()
    dt_string = now.strftime("%d-%m-%Y %H-%M-%S")
    log_dir = os.path.join('logs', dt_string)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # Model
    print('==> Building model..')
    if args.model == 'transformer':
        net = HybridResNetTransformer(in_channels=15, embed_dim=512, depth=4, num_heads=8, out_dim=768)
        metric_fc = ArcMarginProduct(768, 87, s=64.0, m=0.5, K=3)
        if use_cuda:
            metric_fc.cuda()
        # Freeze early ResNet layers at the start
        actual_net = net
        actual_net.freeze_backbone_early_layers()
        print('==> Froze early ResNet layers (conv1, bn1, layer1, layer2) for first', args.freeze_epochs, 'epochs')
    else:
        net = proposed_model.proposed_net(3)
        
    if use_cuda:
        net.cuda()
        cudnn.benchmark = True
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f'==> Using {torch.cuda.device_count()} GPUs with DataParallel!')
            net = nn.DataParallel(net)

    num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print ('num_parameters =', num_parameters)

    criterion = nn.CrossEntropyLoss()
    
    if args.model == 'transformer':
        # Optimize both backbone + arcface layers
        optimizer = optim.AdamW([{'params': net.parameters()}, {'params': metric_fc.parameters()}], lr=3e-4, weight_decay=1e-4)
        scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epoch)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, 50)

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f'==> Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cuda' if use_cuda else 'cpu')
        
        # Load model weights
        if hasattr(net, 'module'):
            net.module.load_state_dict(ckpt['net'])
        else:
            net.load_state_dict(ckpt['net'])
        
        # Load ArcFace weights if transformer
        if args.model == 'transformer' and 'metric_fc' in ckpt:
            metric_fc.load_state_dict(ckpt['metric_fc'])
        
        # Load optimizer and scheduler state
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        
        # If resuming past freeze_epochs, make sure layers are unfrozen
        if args.model == 'transformer' and start_epoch >= args.freeze_epochs:
            actual_net = net.module if hasattr(net, 'module') else net
            actual_net.unfreeze_all()
            print('==> Layers already unfrozen (past freeze epoch)')
        
        # Use the same log_dir from the checkpoint
        if 'log_dir' in ckpt:
            log_dir = ckpt['log_dir']
            if not os.path.isdir(log_dir):
                os.makedirs(log_dir)
            writer = SummaryWriter(log_dir=log_dir)
        
        print(f'==> Resuming from epoch {start_epoch}, best_acc={best_acc:.2f}%')

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            x1, x2, x3, x4, x5 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            if use_cuda:
                targets = targets.cuda()
                x1, x2, x3, x4, x5 = x1.cuda(), x2.cuda(), x3.cuda(), x4.cuda(), x5.cuda()
            x1, x2, x3, x4, x5, targets = Variable(x1), Variable(x2), Variable(x3), Variable(x4), Variable(x5), Variable(targets)
            optimizer.zero_grad()
            
            if args.model == 'transformer':
                x = torch.cat((x1, x2, x3, x4, x5), dim=1)
                
                # Apply Strong Augmentations
                x = augment_tensor(x, std=0.05)
                x, targets_a, targets_b, lam = mixup_data(x, targets, alpha=0.2, use_cuda=use_cuda)
                
                features = net(x)
                outputs = metric_fc(features, targets)
                loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = net(x1, x2, x3, x4, x5)
                loss = criterion(outputs, targets)
            
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients in arcface / transformer
            if args.model == 'transformer':
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                torch.nn.utils.clip_grad_norm_(metric_fc.parameters(), max_norm=5.0)
                
            optimizer.step()

            train_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            
            # For MixUp accuracy, we just measure against the primary target
            if args.model == 'transformer':
                correct += (lam * predicted.eq(targets_a.data).cpu().sum().float() + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            else:
                correct += predicted.eq(targets.data).cpu().sum()

            n_iter = (epoch - 1) * len(trainloader) + batch_idx + 1
            writer.add_scalar('Train/loss', loss.item(), n_iter)

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        writer.add_scalar('Train/acc', correct / len(trainloader.dataset), epoch)
        scheduler.step()


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            x1, x2, x3, x4, x5 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            if use_cuda:
                targets = targets.cuda()
                x1, x2, x3, x4, x5 = x1.cuda(), x2.cuda(), x3.cuda(), x4.cuda(), x5.cuda()
            x1, x2, x3, x4, x5, targets = Variable(x1), Variable(x2), Variable(x3), Variable(x4), Variable(x5), Variable(targets)
            
            if args.model == 'transformer':
                x = torch.cat((x1, x2, x3, x4, x5), dim=1)
                features = net(x)
                outputs = metric_fc(features) # Call without labels to get regular logits at test time
            else:
                outputs = net(x1, x2, x3, x4, x5)
            
            loss = criterion(outputs, targets)

            test_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        writer.add_scalar('Test/Average loss', test_loss / len(testloader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(testloader.dataset), epoch)

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            # Strip 'module.' prefix from DataParallel so weights work on any setup
            net_state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            state = {
                'net': net_state
            }
            if args.model == 'transformer':
                state['metric_fc'] = metric_fc.state_dict()
            torch.save(state, os.path.join(log_dir, 'best.pth'))
            best_acc = acc
            with open(os.path.join(log_dir, 'details.txt'), 'w') as f:
                f.write("{0:.4f}, {1}, lr={2}, batch={3}".format(acc, epoch, args.lr, args.batch_size))
            f.close()


    for epoch in range(start_epoch, start_epoch+args.num_epoch):
        # Unfreeze early layers after freeze_epochs
        if args.model == 'transformer' and epoch == args.freeze_epochs:
            actual_net = net.module if hasattr(net, 'module') else net
            actual_net.unfreeze_all()
            print(f'\n==> Unfreezing all layers at epoch {epoch}')
            num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
            print(f'==> Trainable parameters now: {num_parameters}')
        
        train(epoch)
        test(epoch)
        
        # Save resumable checkpoint every epoch
        net_state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
        last_state = {
            'net': net_state,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch,
            'best_acc': best_acc,
            'log_dir': log_dir
        }
        if args.model == 'transformer':
            last_state['metric_fc'] = metric_fc.state_dict()
        torch.save(last_state, os.path.join(log_dir, 'last.pth'))
