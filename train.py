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
from models.transformer_model import DCT_ViT
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
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
    parser.add_argument('--num_epoch', default=100, type=int, help='Number of epochs')
    parser.add_argument('--model', default='cnn', type=str, choices=['cnn', 'transformer'], help='Model type: cnn or transformer')

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
        net = DCT_ViT(in_channels=3, num_classes=87, embed_dim=256, depth=6, num_heads=8, seq_length=16, num_subbands=5)
    else:
        net = proposed_model.proposed_net(3)
        
    if use_cuda:
        net.cuda()
        cudnn.benchmark = True

    num_parameters = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print ('num_parameters =', num_parameters)

    criterion = nn.CrossEntropyLoss()
    
    if args.model == 'transformer':
        optimizer = optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epoch)
    else:
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = StepLR(optimizer, 50)

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
                outputs = net(x)
            else:
                outputs = net(x1, x2, x3, x4, x5)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.data
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
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
                outputs = net(x)
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
            torch.save(net.state_dict(), os.path.join(log_dir, 'best.pth'))
            best_acc = acc
            with open(os.path.join(log_dir, 'details.txt'), 'w') as f:
                f.write("{0:.4f}, {1}, lr={2}, batch={3}".format(acc, epoch, args.lr, args.batch_size))
            f.close()


    for epoch in range(start_epoch, start_epoch+args.num_epoch):
        train(epoch)
        test(epoch)

