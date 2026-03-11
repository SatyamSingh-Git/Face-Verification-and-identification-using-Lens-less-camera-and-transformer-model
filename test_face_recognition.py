import argparse
import os
import torch.utils.data
from PIL import Image
import numpy as np
from models.proposed_model import proposed_net
from models.transformer_model import DCT_ViT
from my_data_class import Lensless_DCT_offline, Lensless_DCT_offline_noise
from torch.autograd import Variable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for face recognition."
    )

    parser.add_argument(
        '--test_data',
        default="/storage4tb/PycharmProjects/Datasets/lensless_data/test/ymdct_noisy_npy_10px_per_block",
        type=str,
        required=False,
        help='Path to test data')
    parser.add_argument(
        '--weights',
        default="weights/pretrained_weights.pth", type=str,
        required=False,
        help='Path to weights data')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--num_workers', default=3, type=int, help='Number of workers')
    parser.add_argument(
        "--noise_locs",
        type=str,
        default=None,
        help="Path to noise locations .npy file. See data/noise_locations/."
    )
    parser.add_argument('--model', default='cnn', type=str, choices=['cnn', 'transformer'], help='Model type: cnn or transformer')
    

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model == 'transformer':
        net = DCT_ViT(in_channels=3, num_classes=87, embed_dim=256, depth=6, num_heads=8, seq_length=16, num_subbands=5).cuda()
    else:
        net = proposed_net(3).cuda()
        
    checkpoint = torch.load(args.weights)
    net.load_state_dict(checkpoint)
    net.eval()

    if args.noise_locs is not None:
        positions = np.load(args.noise_locs)
        test_data_load = Lensless_DCT_offline_noise(args.test_data, positions)
    else:
        test_data_load = Lensless_DCT_offline(args.test_data)
    
    test_data_loader = torch.utils.data.DataLoader(test_data_load, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    correct = 0.0
    cnt = 0
    total = 0
    for i, (data, targets) in enumerate(test_data_loader):
        x1, x2, x3, x4, x5 = data[0].cuda(), data[1].cuda(), data[2].cuda(), data[3].cuda(), data[4].cuda()
        targets = targets.cuda()
        x1, x2, x3, x4, x5, targets = Variable(x1), Variable(x2), Variable(x3), Variable(x4), Variable(x5), Variable(
            targets)
            
        if args.model == 'transformer':
            x = torch.cat((x1, x2, x3, x4, x5), dim=1)
            outputs = net(x)
        else:
            outputs = net(x1, x2, x3, x4, x5)

        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        cnt += 1
        print ('Processed epoch', cnt, '/', len(test_data_loader))
    correct = correct.data.cpu().numpy()
    print ('Accuracy is', correct, '/', total, '=', correct*100/total)

