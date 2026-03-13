import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from models.proposed_model import proposed_net
from models.transformer_model import HybridResNetTransformer
from scipy.spatial.distance import cosine
import json
from sklearn import metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for face recognition."
    )

    parser.add_argument(
        '--test_data',
        default="/storage4tb/PycharmProjects/Datasets/lensless_data/test/ymdct_npy",
        type=str,
        help='Path to test data')
    parser.add_argument(
        '--pairs',
        default="data/verification_pairs.txt",
        type=str,
        help='Path to .txt file containing verification pairs')
    parser.add_argument(
        '--weights',
        default="weights/pretrained_weights.pth",
        type=str,
        help='Path to weights data')
    parser.add_argument(
        '--out_file',
        default="results.json",
        type=str,
        help='Path to output .json file containing results')
    parser.add_argument(
        '--thresh',
        default=0.5,
        type=float,
        help='Threshold value for cosine similarity used for deciding between match or no-match.')
    parser.add_argument('--model', default='cnn', type=str, choices=['cnn', 'transformer'], help='Model type: cnn or transformer')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()    

    if args.model == 'transformer':
        net = HybridResNetTransformer(in_channels=15, embed_dim=512, depth=4, num_heads=8).cuda()
        checkpoint = torch.load(args.weights)
        if isinstance(checkpoint, dict) and 'net' in checkpoint:
            net.load_state_dict(checkpoint['net'])
        else:
            net.load_state_dict(checkpoint) # fallback
        # The HybridResNetTransformer natively outputs 512D embeddings!
        if torch.cuda.device_count() > 1:
            print(f'==> Using {torch.cuda.device_count()} GPUs!')
            net = nn.DataParallel(net)
    else:
        net = proposed_net(3).cuda()
        net.load_state_dict(torch.load(args.weights))
        net.classifier = net.classifier[:-3]
        if torch.cuda.device_count() > 1:
            print(f'==> Using {torch.cuda.device_count()} GPUs!')
            net = nn.DataParallel(net)
    # print (net.classifier)
    net.eval()

    correct = 0
    cnt = 0
    true_labels = []
    total = sum(len(files) for _, _, files in os.walk(args.test_data))
    with torch.no_grad():
        pred_scores = []
        with open(args.pairs) as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                i += 1
                line = line.replace("\n", "")
                line = line.split(" ")
                line_1 = line[0].split('.')[0]
                line_2 = line[1].split('.')[0]
                label = int(line[2])
                true_labels.append(label)
                
                n_arr_1 = torch.from_numpy(np.load(os.path.join(args.test_data, line_1 + '.npy'))).float().to('cuda')
                n_arr_2 = torch.from_numpy(np.load(os.path.join(args.test_data, line_2 + '.npy'))).float().to('cuda')
                x1 = n_arr_1[0:3, :, :].unsqueeze(0)
                x2 = n_arr_1[3:6, :, :].unsqueeze(0)
                x3 = n_arr_1[6:9, :, :].unsqueeze(0)
                x4 = n_arr_1[9:12, :, :].unsqueeze(0)
                x5 = n_arr_1[12:15, :, :].unsqueeze(0)
                if args.model == 'transformer':
                    x = torch.cat((x1, x2, x3, x4, x5), dim=1)
                    output_1 = net(x, tta=True).cpu().detach().numpy()
                else:
                    output_1 = net(x1, x2, x3, x4, x5).cpu().detach().numpy()
                output_1 = output_1.reshape(-1)
                
                x1 = n_arr_2[0:3, :, :].unsqueeze(0)
                x2 = n_arr_2[3:6, :, :].unsqueeze(0)
                x3 = n_arr_2[6:9, :, :].unsqueeze(0)
                x4 = n_arr_2[9:12, :, :].unsqueeze(0)
                x5 = n_arr_2[12:15, :, :].unsqueeze(0)
                if args.model == 'transformer':
                    x = torch.cat((x1, x2, x3, x4, x5), dim=1)
                    output_2 = net(x, tta=True).cpu().detach().numpy()
                else:
                    output_2 = net(x1, x2, x3, x4, x5).cpu().detach().numpy()
                output_2 = output_2.reshape(-1)

                score = 1 - cosine(output_1, output_2)
                pred_scores.append(score)
                if score >= args.thresh:
                    pred = 1
                else:
                    pred = 0
                if pred == label:
                    correct += 1
            cnt += 1
            if cnt % 1000 == 0:
                print(cnt, '/', len(lines), 'correct =', correct)
        print (f'Accuracy = {correct}/{len(lines)} = {correct/len(lines)}')
    
    auc = round(metrics.roc_auc_score(true_labels, pred_scores), 4)
    print ('AUC =', auc)
    
    output_dict = {
        'true_labels': true_labels,
        'pred_scores': pred_scores,
        'thresh': args.thresh,
        'acc': correct/len(lines)
    }

    json = json.dumps(output_dict)
    f = open(args.out_file,"w")
    f.write(json)
    f.close()


