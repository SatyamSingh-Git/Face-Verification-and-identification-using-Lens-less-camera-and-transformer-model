#!/usr/bin/env python
"""
Generate evaluation graphs for the best face recognition model.

Produces:
  1. Confusion Matrix (heatmap)
  2. ROC Curve (with AUC)
  3. Per-Class Accuracy (bar chart)
  4. Learning Curves (train/test loss + accuracy from TensorBoard logs)
  5. Top-K Accuracy curve

Usage:
  python generate_graphs.py --weights saved_models/t1_ls_swa/best.pth \
                            --test_data /path/to/test/ymdct_npy \
                            --output_dir graphs/

On Trinity:
  python generate_graphs.py --weights saved_models/t1_ls_swa/best.pth \
                            --test_data /trinity/home/satyam231220054/lensless_data/test/ymdct_npy \
                            --log_dir logs/  --output_dir graphs/
"""

import argparse
import os
import sys
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable

# Add parent dir to path so we can import project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import my_data_class
from models.transformer_model import HybridResNetTransformer
from models.arcface import ArcMarginProduct

# ── Plotting imports ──────────────────────────────────────────────────────────
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    from sklearn.metrics import (
        confusion_matrix, roc_curve, auc,
        classification_report, precision_recall_fscore_support
    )
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scikit-learn not found. Some plots will be skipped.")

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False


def parse_args():
    parser = argparse.ArgumentParser(description='Generate evaluation graphs')
    parser.add_argument('--weights', required=True, type=str,
                        help='Path to best.pth checkpoint')
    parser.add_argument('--test_data', required=True, type=str,
                        help='Path to test dataset (ymdct_npy)')
    parser.add_argument('--output_dir', default='graphs', type=str,
                        help='Output directory for graphs')
    parser.add_argument('--log_dir', default='', type=str,
                        help='TensorBoard log directory for learning curves')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    # Model config (must match training)
    parser.add_argument('--backbone', default='resnet18', type=str)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--use_gem', action='store_true')
    parser.add_argument('--embed_dropout', default=0.0, type=float)
    parser.add_argument('--arcface_k', default=3, type=int)
    parser.add_argument('--use_tta', action='store_true', default=True,
                        help='Use 5-view TTA (default: True)')
    parser.add_argument('--no_tta', action='store_true',
                        help='Disable TTA')
    return parser.parse_args()


def load_model(args, device):
    """Load model and ArcFace head from checkpoint."""
    net = HybridResNetTransformer(
        in_channels=15, embed_dim=512, depth=4, num_heads=8, out_dim=768,
        backbone=args.backbone, input_size=args.input_size,
        use_gem=args.use_gem, embed_dropout=args.embed_dropout
    )
    metric_fc = ArcMarginProduct(768, 87, s=40.0, m=0.35, K=args.arcface_k)

    checkpoint = torch.load(args.weights, map_location=device)
    net.load_state_dict(checkpoint['net'])
    if 'metric_fc' in checkpoint:
        metric_fc.load_state_dict(checkpoint['metric_fc'])

    net = net.to(device)
    metric_fc = metric_fc.to(device)
    net.eval()
    metric_fc.eval()
    return net, metric_fc


def run_inference(net, metric_fc, testloader, device, use_tta=True):
    """Run inference on test set, return predictions, labels, probabilities."""
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            x1, x2, x3, x4, x5 = inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
            x1 = x1.to(device)
            x2 = x2.to(device)
            x3 = x3.to(device)
            x4 = x4.to(device)
            x5 = x5.to(device)
            targets = targets.to(device)

            x = torch.cat((x1, x2, x3, x4, x5), dim=1)
            features = net(x, tta=use_tta)
            outputs = metric_fc(features)  # No labels → regular logits

            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_features.append(features.cpu().numpy())

            sys.stdout.write(f'\r  Inference: {batch_idx+1}/{len(testloader)} batches')
            sys.stdout.flush()

    print()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    all_features = np.concatenate(all_features, axis=0)
    return all_preds, all_labels, all_probs, all_features


def plot_confusion_matrix(labels, preds, output_dir, num_classes=87):
    """Plot confusion matrix heatmap."""
    if not HAS_SKLEARN:
        print("  [SKIP] Confusion matrix (no sklearn)")
        return

    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    acc = np.trace(cm) / cm.sum() * 100

    fig, ax = plt.subplots(figsize=(16, 14))

    if HAS_SEABORN:
        sns.heatmap(cm, cmap='Blues', ax=ax, cbar_kws={'shrink': 0.8},
                    xticklabels=5, yticklabels=5, linewidths=0)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        fig.colorbar(im, ax=ax, shrink=0.8)

    ax.set_xlabel('Predicted Label', fontsize=14)
    ax.set_ylabel('True Label', fontsize=14)
    ax.set_title(f'Confusion Matrix — {acc:.2f}% Accuracy ({num_classes} classes)', fontsize=16)

    plt.tight_layout()
    path = os.path.join(output_dir, 'confusion_matrix.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [SAVED] {path}')


def plot_roc_curve(labels, probs, output_dir, num_classes=87):
    """Plot one-vs-rest ROC curve (macro-average + individual)."""
    if not HAS_SKLEARN:
        print("  [SKIP] ROC curve (no sklearn)")
        return

    from sklearn.preprocessing import label_binarize

    # Binarize labels for one-vs-rest
    y_bin = label_binarize(labels, classes=range(num_classes))

    # Compute macro-averaged ROC
    fpr_macro, tpr_macro, _ = roc_curve(y_bin.ravel(), probs.ravel())
    roc_auc_macro = auc(fpr_macro, tpr_macro)

    # Per-class ROC (compute AUCs)
    per_class_auc = []
    for i in range(num_classes):
        if y_bin[:, i].sum() > 0:
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i])
            per_class_auc.append(auc(fpr_i, tpr_i))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot a few individual class curves (worst 5)
    auc_list = []
    for i in range(num_classes):
        if y_bin[:, i].sum() > 0:
            fpr_i, tpr_i, _ = roc_curve(y_bin[:, i], probs[:, i])
            auc_list.append((auc(fpr_i, tpr_i), i, fpr_i, tpr_i))

    auc_list.sort(key=lambda x: x[0])

    # Plot 5 worst performing classes
    for auc_val, cls_id, fpr_i, tpr_i in auc_list[:5]:
        ax.plot(fpr_i, tpr_i, alpha=0.3, linewidth=1,
                label=f'Class {cls_id} (AUC={auc_val:.3f})')

    # Macro average
    ax.plot(fpr_macro, tpr_macro, 'b-', linewidth=2.5,
            label=f'Macro-Average (AUC={roc_auc_macro:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title(f'ROC Curve — Macro AUC: {roc_auc_macro:.4f}', fontsize=16)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'roc_curve.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [SAVED] {path}')

    # Print mean per-class AUC
    mean_auc = np.mean(per_class_auc)
    print(f'  Mean per-class AUC: {mean_auc:.4f}')


def plot_per_class_accuracy(labels, preds, output_dir, num_classes=87):
    """Bar chart of per-class accuracy, sorted."""
    if not HAS_SKLEARN:
        print("  [SKIP] Per-class accuracy (no sklearn)")
        return

    cm = confusion_matrix(labels, preds, labels=range(num_classes))
    per_class_acc = np.diag(cm) / (cm.sum(axis=1) + 1e-10) * 100

    # Sort by accuracy
    sorted_idx = np.argsort(per_class_acc)

    fig, ax = plt.subplots(figsize=(18, 8))
    colors = ['#e74c3c' if a < 90 else '#f39c12' if a < 95 else '#2ecc71'
              for a in per_class_acc[sorted_idx]]

    ax.bar(range(num_classes), per_class_acc[sorted_idx], color=colors, edgecolor='none')
    ax.set_xlabel('Class (sorted by accuracy)', fontsize=14)
    ax.set_ylabel('Accuracy (%)', fontsize=14)
    ax.set_title(f'Per-Class Accuracy — Mean: {per_class_acc.mean():.2f}%', fontsize=16)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3, label='100%')
    ax.axhline(y=per_class_acc.mean(), color='blue', linestyle='--', alpha=0.5,
               label=f'Mean: {per_class_acc.mean():.1f}%')
    ax.set_ylim([0, 105])
    ax.legend(fontsize=12)
    ax.grid(axis='y', alpha=0.3)

    # Annotate worst classes
    for i in range(min(5, num_classes)):
        cls_id = sorted_idx[i]
        acc = per_class_acc[cls_id]
        ax.annotate(f'C{cls_id}: {acc:.0f}%', xy=(i, acc + 1),
                    fontsize=7, ha='center', color='red')

    plt.tight_layout()
    path = os.path.join(output_dir, 'per_class_accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [SAVED] {path}')

    # Print worst 10 classes
    print(f'\n  --- Worst 10 Classes ---')
    for i in range(min(10, num_classes)):
        cls_id = sorted_idx[i]
        n_samples = cm[cls_id].sum()
        n_correct = cm[cls_id, cls_id]
        print(f'    Class {cls_id:3d}: {per_class_acc[cls_id]:5.1f}% ({n_correct}/{n_samples})')


def plot_topk_accuracy(labels, probs, output_dir, max_k=10):
    """Plot Top-K accuracy for K=1..max_k."""
    topk_accs = []
    for k in range(1, max_k + 1):
        topk_preds = np.argsort(probs, axis=1)[:, -k:]
        correct = sum(1 for i, lbl in enumerate(labels) if lbl in topk_preds[i])
        topk_accs.append(correct / len(labels) * 100)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, max_k + 1), topk_accs, 'bo-', linewidth=2, markersize=8)

    for k, acc in enumerate(topk_accs, 1):
        ax.annotate(f'{acc:.1f}%', xy=(k, acc), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=9)

    ax.set_xlabel('K', fontsize=14)
    ax.set_ylabel('Top-K Accuracy (%)', fontsize=14)
    ax.set_title('Top-K Accuracy', fontsize=16)
    ax.set_xticks(range(1, max_k + 1))
    ax.set_ylim([min(topk_accs) - 2, 101])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'topk_accuracy.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [SAVED] {path}')


def plot_learning_curves(log_dir, output_dir):
    """Plot learning curves from TensorBoard logs."""
    if not HAS_TB:
        print("  [SKIP] Learning curves (tensorboard not available)")
        return
    if not log_dir or not os.path.isdir(log_dir):
        print("  [SKIP] Learning curves (no log_dir provided)")
        return

    # Find the most recent TensorBoard event file
    event_dirs = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.startswith('events.out.tfevents'):
                event_dirs.append(root)
                break

    if not event_dirs:
        print("  [SKIP] Learning curves (no TensorBoard events found)")
        return

    # Use the most recent directory
    event_dir = sorted(event_dirs)[-1]
    print(f'  Reading TensorBoard logs from: {event_dir}')

    ea = EventAccumulator(event_dir)
    ea.Reload()

    tags = ea.Tags().get('scalars', [])
    print(f'  Available tags: {tags}')

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Loss curves
    ax = axes[0]
    for tag in ['Train/Average loss', 'Test/Average loss']:
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            label = 'Train Loss' if 'Train' in tag else 'Test Loss'
            ax.plot(steps, values, label=label, linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Test Loss', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Accuracy curve
    ax = axes[1]
    for tag in ['Test/Accuracy', 'Train/Accuracy']:
        if tag in tags:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value * 100 for e in events]  # Convert to %
            label = 'Test Accuracy' if 'Test' in tag else 'Train Accuracy'
            ax.plot(steps, values, label=label, linewidth=1.5)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Test Accuracy', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, 'learning_curves.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  [SAVED] {path}')


def print_summary(labels, preds, probs):
    """Print text summary of results."""
    total = len(labels)
    correct = (preds == labels).sum()
    acc = correct / total * 100

    # Top-5
    top5_preds = np.argsort(probs, axis=1)[:, -5:]
    top5_correct = sum(1 for i, lbl in enumerate(labels) if lbl in top5_preds[i])
    top5_acc = top5_correct / total * 100

    print(f'\n{"="*60}')
    print(f'  EVALUATION SUMMARY')
    print(f'{"="*60}')
    print(f'  Total samples  : {total}')
    print(f'  Correct        : {correct}')
    print(f'  Top-1 Accuracy : {acc:.2f}%')
    print(f'  Top-5 Accuracy : {top5_acc:.2f}%')
    print(f'  Errors         : {total - correct}')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    args = parse_args()
    use_tta = not args.no_tta

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    print('\n[1/5] Loading model...')
    net, metric_fc = load_model(args, device)

    # Load test data
    print('[2/5] Loading test data...')
    testset = my_data_class.Lensless_DCT_offline(args.test_data)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers
    )
    print(f'  Test samples: {len(testset)}, Batches: {len(testloader)}')

    # Run inference
    print(f'[3/5] Running inference (TTA={use_tta})...')
    preds, labels, probs, features = run_inference(
        net, metric_fc, testloader, device, use_tta=use_tta
    )

    # Print summary
    print_summary(labels, preds, probs)

    # Generate graphs
    print('[4/5] Generating graphs...')
    plot_confusion_matrix(labels, preds, args.output_dir)
    plot_roc_curve(labels, probs, args.output_dir)
    plot_per_class_accuracy(labels, preds, args.output_dir)
    plot_topk_accuracy(labels, probs, args.output_dir)

    # Learning curves (optional)
    print('[5/5] Learning curves...')
    plot_learning_curves(args.log_dir, args.output_dir)

    print(f'\n✓ All graphs saved to: {args.output_dir}/')
