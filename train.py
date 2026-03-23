import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.swa_utils import AveragedModel, update_bn, SWALR
import my_data_class
import math
import copy

# TPU Support
try:
    import torch_xla.core.xla_model as xm
    HAS_XLA = True
except ImportError:
    HAS_XLA = False

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
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np
import random

def mixup_data(x, y, alpha=0.2, use_cuda=True):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def cutmix_data(x, y, alpha=1.0):
    """CutMix: cut a patch from one image and paste onto another."""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # Generate random bounding box
    W, H = x.size(2), x.size(3)
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_ratio)
    cut_h = int(H * cut_ratio)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    mixed_x = x.clone()
    mixed_x[:, :, x1:x2, y1:y2] = x[index, :, x1:x2, y1:y2]

    # Adjust lambda to the actual area ratio
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))

    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

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

def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6, base_lr=3e-4):
    """Linear warmup for `warmup_epochs`, then cosine decay with a minimum LR floor."""
    min_lr_multiplier = eta_min / base_lr  # Multiplier that gives eta_min absolute LR
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        else:
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Clamp to minimum multiplier so LR never collapses to 0
            return max(cosine_decay, min_lr_multiplier)
    return LambdaLR(optimizer, lr_lambda)


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
    parser.add_argument('--eta_min', default=1e-6, type=float, help='Minimum learning rate floor for cosine scheduler (prevents LR collapse and NaN loss)')
    # Ablation flags
    parser.add_argument('--seed', default=42, type=int, help='Random seed for reproducibility')
    parser.add_argument('--no_mixup', action='store_true', help='Disable Mixup augmentation')
    parser.add_argument('--arcface_m', default=0.5, type=float, help='ArcFace margin')
    parser.add_argument('--arcface_s', default=64.0, type=float, help='ArcFace scale')
    parser.add_argument('--arcface_k', default=3, type=int, help='ArcFace sub-center count')
    parser.add_argument('--backbone', default='resnet18', type=str, choices=['resnet18', 'resnet34'],
                        help='CNN backbone: resnet18 (default) or resnet34')

    # Tier-1 improvements
    parser.add_argument('--label_smoothing', default=0.1, type=float, help='Label smoothing factor (0 = off, 0.1 recommended)')
    parser.add_argument('--use_swa', action='store_true', help='Enable Stochastic Weight Averaging')
    parser.add_argument('--swa_start_frac', default=0.75, type=float, help='Start SWA at this fraction of total epochs')
    parser.add_argument('--use_cutmix', action='store_true', help='Enable CutMix (50/50 with Mixup per batch)')
    parser.add_argument('--use_warm_restarts', action='store_true', help='Use CosineAnnealingWarmRestarts instead of single cosine decay')
    parser.add_argument('--restart_t0', default=50, type=int, help='T_0 for warm restarts (first cycle length in epochs)')
    parser.add_argument('--restart_tmult', default=2, type=int, help='T_mult for warm restarts (cycle length multiplier)')

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if HAS_XLA:
        device = xm.xla_device()
        is_tpu = True
        use_cuda = False
        print(f'==> Using TPU: {device}')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        is_tpu = False
        use_cuda = True
        print(f'==> Using GPU(s): {torch.cuda.device_count()}')
    else:
        device = torch.device('cpu')
        is_tpu = False
        use_cuda = False
        print('==> Using CPU')

    # AMP is permanently disabled — it caused NaN loss instability across all runs.
    # The original stable training (v1, 94.4%) ran without AMP. FP32 is used throughout.
    scaler = None

    best_acc = 0
    start_epoch = 0

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
    print(f'==> Seed: {args.seed}')

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
        net = HybridResNetTransformer(in_channels=15, embed_dim=512, depth=4, num_heads=8, out_dim=768, backbone=args.backbone)
        metric_fc = ArcMarginProduct(768, 87, s=args.arcface_s, m=args.arcface_m, K=args.arcface_k)
        print(f'==> Backbone: {args.backbone}')
        print(f'==> ArcFace config: s={args.arcface_s}, m={args.arcface_m}, K={args.arcface_k}')
        print(f'==> Mixup: {"ON" if not args.no_mixup else "OFF"}')
        if is_tpu or use_cuda:
            metric_fc.to(device)
        # Freeze early ResNet layers at the start
        actual_net = net
        actual_net.freeze_backbone_early_layers()
        print('==> Froze early ResNet layers (conv1, bn1, layer1, layer2) for first', args.freeze_epochs, 'epochs')
    else:
        net = proposed_model.proposed_net(3)
        
    net.to(device)
    if use_cuda:
        cudnn.benchmark = True
        # Multi-GPU support
        if torch.cuda.device_count() > 1:
            print(f'==> Using {torch.cuda.device_count()} GPUs with DataParallel!')
            net = nn.DataParallel(net)

    num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print ('num_parameters =', num_parameters)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    if args.label_smoothing > 0:
        print(f'==> Label Smoothing: {args.label_smoothing}')
    
    if args.model == 'transformer':
        # Optimize both backbone + arcface layers
        optimizer = optim.AdamW([{'params': net.parameters()}, {'params': metric_fc.parameters()}], lr=3e-4, weight_decay=1e-4)
        if args.use_warm_restarts:
            # Warmup for warmup_epochs, then CosineAnnealingWarmRestarts
            scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epoch, eta_min=args.eta_min, base_lr=3e-4)
            # We'll switch to warm restarts after warmup in the training loop
            print(f'==> Scheduler: Warmup({args.warmup_epochs}ep) + CosineWarmRestarts(T0={args.restart_t0}, Tmult={args.restart_tmult})')
        else:
            scheduler = get_warmup_cosine_scheduler(optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.num_epoch, eta_min=args.eta_min, base_lr=3e-4)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, 50)

    # Resume from checkpoint if specified
    if args.resume is not None:
        print(f'==> Resuming from checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location=device)
        
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

    # --- SWA Setup ---
    swa_model = None
    swa_scheduler = None
    swa_start_epoch = int(args.num_epoch * args.swa_start_frac) if args.use_swa else args.num_epoch + 1
    if args.use_swa and args.model == 'transformer':
        swa_model = AveragedModel(net)
        swa_scheduler = SWALR(optimizer, swa_lr=1e-5, anneal_epochs=5)
        print(f'==> SWA: ON (starts at epoch {swa_start_epoch})')

    # --- Warm Restarts scheduler (created after warmup) ---
    warm_restart_scheduler = None
    if args.use_warm_restarts and args.model == 'transformer':
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        # Will be activated after warmup_epochs

    # --- CutMix flag ---
    if args.use_cutmix:
        print('==> CutMix: ON (50/50 with Mixup per batch)')

    # Training
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            x1, x2, x3, x4, x5 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            if is_tpu or use_cuda:
                targets = targets.to(device)
                x1, x2, x3, x4, x5 = x1.to(device), x2.to(device), x3.to(device), x4.to(device), x5.to(device)
            x1, x2, x3, x4, x5, targets = Variable(x1), Variable(x2), Variable(x3), Variable(x4), Variable(x5), Variable(targets)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda', enabled=False):  # AMP disabled: caused NaN instability
                if args.model == 'transformer':
                    x = torch.cat((x1, x2, x3, x4, x5), dim=1)
                    
                    # Apply Strong Augmentations
                    x = augment_tensor(x, std=0.05)
                    
                    if not args.no_mixup:
                        # Choose Mixup or CutMix (50/50 per batch)
                        if args.use_cutmix and random.random() > 0.5:
                            x, targets_a, targets_b, lam = cutmix_data(x, targets, alpha=1.0)
                        else:
                            x, targets_a, targets_b, lam = mixup_data(x, targets, alpha=0.2)
                        features = net(x)
                        outputs = metric_fc(features, targets)
                        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                    else:
                        # Mixup/CutMix OFF
                        features = net(x)
                        outputs = metric_fc(features, targets)
                        loss = criterion(outputs, targets)
                        lam = 1.0
                        targets_a = targets
                        targets_b = targets
                else:
                    outputs = net(x1, x2, x3, x4, x5)
                    loss = criterion(outputs, targets)
            
            if scaler is not None:
                scaler.scale(loss).backward()
                if args.model == 'transformer':
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(metric_fc.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.model == 'transformer':
                    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
                    torch.nn.utils.clip_grad_norm_(metric_fc.parameters(), max_norm=5.0)
                if is_tpu:
                    xm.optimizer_step(optimizer)
                    xm.mark_step()
                else:
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


        writer.add_scalar('Train/acc', correct / len(trainloader.dataset), epoch)
        print('  Train | Loss: %.3f | Acc: %.3f%% (%d/%d)'
              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # Step the appropriate scheduler
        if args.use_swa and epoch >= swa_start_epoch:
            swa_scheduler.step()
        elif args.use_warm_restarts and warm_restart_scheduler is not None:
            warm_restart_scheduler.step()
        else:
            scheduler.step()


    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            x1, x2, x3, x4, x5 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            if is_tpu or use_cuda:
                targets = targets.to(device)
                x1, x2, x3, x4, x5 = x1.to(device), x2.to(device), x3.to(device), x4.to(device), x5.to(device)
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


        print('  Test  | Loss: %.3f | Acc: %.3f%% (%d/%d)'
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

        # Switch to warm restarts scheduler after warmup
        if args.use_warm_restarts and args.model == 'transformer' and epoch == args.warmup_epochs and warm_restart_scheduler is None:
            from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
            warm_restart_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.restart_t0, T_mult=args.restart_tmult, eta_min=args.eta_min)
            print(f'\n==> Switching to CosineAnnealingWarmRestarts (T0={args.restart_t0}, Tmult={args.restart_tmult})')
        
        train(epoch)
        test(epoch)

        # Update SWA model after swa_start_epoch
        if args.use_swa and swa_model is not None and epoch >= swa_start_epoch:
            swa_model.update_parameters(net)
        
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

    # --- Save SWA model at end of training ---
    if args.use_swa and swa_model is not None:
        print('\n==> Updating SWA BatchNorm statistics...')
        # Manual BN update — our data loader returns ((x1..x5), label) which
        # torch.optim.swa_utils.update_bn cannot handle directly.
        swa_model.train()
        with torch.no_grad():
            # Reset BN running stats
            for module in swa_model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.running_mean.zero_()
                    module.running_var.fill_(1)
                    module.num_batches_tracked.zero_()
            # Forward pass through training data to recompute BN stats
            for inputs, _ in trainloader:
                x1, x2, x3, x4, x5 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
                if use_cuda:
                    x1, x2, x3, x4, x5 = x1.to(device), x2.to(device), x3.to(device), x4.to(device), x5.to(device)
                x = torch.cat((x1, x2, x3, x4, x5), dim=1)
                swa_model(x)
        swa_state = swa_model.module.state_dict()
        swa_save = {'net': swa_state}
        if args.model == 'transformer':
            swa_save['metric_fc'] = metric_fc.state_dict()
        swa_path = os.path.join(log_dir, 'swa_best.pth')
        torch.save(swa_save, swa_path)
        print(f'==> SWA model saved: {swa_path}')
