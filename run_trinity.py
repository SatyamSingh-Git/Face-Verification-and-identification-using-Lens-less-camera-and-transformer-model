#!/trinity/home/satyam231220054/satyam/bin/python
# coding: utf-8
"""
Standalone training + evaluation script for Trinity/On-demand (RTX A6000).
Converted from transformer_colab_run_new.ipynb with all path and logic bugs fixed.

Usage:
    python run_trinity.py                         # Run with defaults below
    python run_trinity.py --skip_training         # Skip training, just evaluate
    python run_trinity.py --batch_size 256        # Override batch size
    python run_trinity.py --num_epoch 150         # Override epochs

All training hyperparameters can be overridden via command-line flags.
See the "USER CONFIG (defaults)" section and argparse block below.
"""

import os
import sys
import glob
import json
import shutil
import subprocess
import argparse
from pathlib import Path

import numpy as np

# =============================================================================
# USER CONFIG (defaults — override any of these via command-line flags)
# =============================================================================
PYTHON_EXEC = "/trinity/home/satyam231220054/satyam/bin/python"

REPO_PATH = Path(
    "/trinity/home/satyam231220054/"
    "Face-Verification-and-identification-using-Lens-less-camera-and-transformer-model/"
    "lensless_face_recognition"
)

DEFAULT_TRAIN_DATA = "/trinity/home/satyam231220054/lensless_data/train/ymdct_npy"
DEFAULT_TEST_DATA  = "/trinity/home/satyam231220054/lensless_data/test/ymdct_npy"

# =============================================================================
# COMMAND-LINE ARGUMENT PARSER
# =============================================================================
# ── This is where you customise your training run ──
# You can change any of these defaults here, OR override them on the command
# line when you submit the job, for example:
#     python run_trinity.py --batch_size 256 --num_epoch 150 --arcface_m 0.4

def build_parser():
    parser = argparse.ArgumentParser(
        description="Trinity standalone training + evaluation script"
    )

    # Paths
    parser.add_argument("--train_data", type=str, default=DEFAULT_TRAIN_DATA,
                        help="Path to training data")
    parser.add_argument("--test_data",  type=str, default=DEFAULT_TEST_DATA,
                        help="Path to test data")

    # Training hyperparameters
    parser.add_argument("--model",       type=str,   default="transformer")
    parser.add_argument("--batch_size",  type=int,   default=64,
                        help="Training batch size (RTX A6000 can handle 128–256)")
    parser.add_argument("--num_epoch",   type=int,   default=150,
                        help="Number of training epochs")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--eta_min",     type=float, default=1e-6,
                        help="Minimum LR floor for cosine scheduler")
    parser.add_argument("--num_workers", type=int,   default=4,
                        help="DataLoader workers (A6000 can use 4+)")

    # ArcFace config — these gave the best result (94.73%) in our experiments
    parser.add_argument("--arcface_m",   type=float, default=0.4,
                        help="ArcFace margin")
    parser.add_argument("--arcface_s",   type=float, default=40.0,
                        help="ArcFace scale")
    parser.add_argument("--arcface_k",   type=int,   default=3,
                        help="ArcFace sub-center count")

    # Experiment naming
    parser.add_argument("--save_name",   type=str,   default="exp_v2_baseline_s42",
                        help="Name for saved_models subdirectory")

    # Test config
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--no_tta",          action="store_true",
                        help="Disable Test-Time Augmentation")

    # Tier-1 improvements
    parser.add_argument("--label_smoothing", type=float, default=0.1,
                        help="Label smoothing factor (0 = off)")
    parser.add_argument("--use_swa",          action="store_true",
                        help="Enable Stochastic Weight Averaging")
    parser.add_argument("--swa_start_frac",  type=float, default=0.75,
                        help="Start SWA at this fraction of total epochs")
    parser.add_argument("--use_cutmix",       action="store_true",
                        help="Enable CutMix (50/50 with Mixup per batch)")
    parser.add_argument("--use_warm_restarts",action="store_true",
                        help="Use CosineAnnealingWarmRestarts scheduler")
    parser.add_argument("--restart_t0",      type=int, default=50,
                        help="T_0 for warm restarts")
    parser.add_argument("--restart_tmult",   type=int, default=2,
                        help="T_mult for warm restarts")

    # Tier-2: backbone selection
    parser.add_argument("--backbone",         type=str, default="resnet18",
                        choices=["resnet18", "resnet34"],
                        help="CNN backbone: resnet18 (default) or resnet34")

    # Tier-3 improvements
    parser.add_argument("--input_size",       type=int, default=224,
                        help="Input resolution (224 default, try 112)")
    parser.add_argument("--use_gem",          action="store_true",
                        help="Use GeM pooling instead of mean pooling")
    parser.add_argument("--embed_dropout",    type=float, default=0.0,
                        help="Dropout before projection head (0=off)")

    # Workflow control
    parser.add_argument("--skip_training",   action="store_true",
                        help="Skip training; only run evaluation on existing checkpoint")
    parser.add_argument("--do_install",      action="store_true",
                        help="Run pip install steps (disabled by default — Trinity blocks internet)")
    parser.add_argument("--batch_experiments", action="store_true",
                        help="Run all batch experiments sequentially (overnight mode)")
    parser.add_argument("--batch_tier1",     action="store_true",
                        help="Run Tier-1 accuracy improvement experiments")
    parser.add_argument("--batch_tier2",     action="store_true",
                        help="Run Tier-2 ResNet34 backbone experiments")
    parser.add_argument("--batch_tier3",     action="store_true",
                        help="Run Tier-3 resolution/GeM/dropout experiments")

    return parser


# =============================================================================
# BATCH EXPERIMENT DEFINITIONS
# =============================================================================
# Each experiment overrides specific args. Everything else uses CLI defaults.
BATCH_EXPERIMENTS = [
    {
        "name": "exp_200ep",
        "desc": "Experiment 1/4: Extended training (200 epochs)",
        "overrides": {"num_epoch": 200},
    },
    {
        "name": "exp_m035_200ep",
        "desc": "Experiment 2/4: Softer margin m=0.35 (200 epochs)",
        "overrides": {"num_epoch": 200, "arcface_m": 0.35},
    },
    {
        "name": "exp_k5_200ep",
        "desc": "Experiment 3/4: More sub-centers K=5 (200 epochs)",
        "overrides": {"num_epoch": 200, "arcface_k": 5},
    },
    {
        "name": "exp_m035_k5_200ep",
        "desc": "Experiment 4/4: Combined m=0.35 + K=5 (200 epochs)",
        "overrides": {"num_epoch": 200, "arcface_m": 0.35, "arcface_k": 5},
    },
]

# Tier-1 improvement experiments (stacking approach)
BATCH_EXPERIMENTS_T1 = [
    {
        "name": "t1_label_smooth",
        "desc": "T1-Exp 1/4: Label Smoothing 0.1 (250ep, m=0.35)",
        "overrides": {
            "num_epoch": 250, "arcface_m": 0.35,
            "label_smoothing": 0.1,
        },
    },
    {
        "name": "t1_ls_swa",
        "desc": "T1-Exp 2/4: + Stochastic Weight Averaging",
        "overrides": {
            "num_epoch": 250, "arcface_m": 0.35,
            "label_smoothing": 0.1, "use_swa": True,
        },
    },
    {
        "name": "t1_ls_swa_cutmix",
        "desc": "T1-Exp 3/4: + CutMix (50/50 with Mixup)",
        "overrides": {
            "num_epoch": 250, "arcface_m": 0.35,
            "label_smoothing": 0.1, "use_swa": True, "use_cutmix": True,
        },
    },
    {
        "name": "t1_full",
        "desc": "T1-Exp 4/4: Full Tier-1 (+ Warm Restarts)",
        "overrides": {
            "num_epoch": 250, "arcface_m": 0.35,
            "label_smoothing": 0.1, "use_swa": True, "use_cutmix": True,
            "use_warm_restarts": True, "restart_t0": 50, "restart_tmult": 2,
        },
    },
]

# Tier-2: ResNet34 backbone experiments (using best Tier-1 config: LS + SWA)
BATCH_EXPERIMENTS_T2 = [
    {
        "name": "t2_resnet34",
        "desc": "T2-Exp 1/2: ResNet34 + LS + SWA (250ep, m=0.35)",
        "overrides": {
            "num_epoch": 250, "arcface_m": 0.35,
            "label_smoothing": 0.1, "use_swa": True,
            "backbone": "resnet34",
        },
    },
    {
        "name": "t2_resnet34_wr",
        "desc": "T2-Exp 2/2: ResNet34 + LS + SWA + Warm Restarts (250ep, m=0.35)",
        "overrides": {
            "num_epoch": 250, "arcface_m": 0.35,
            "label_smoothing": 0.1, "use_swa": True,
            "use_warm_restarts": True, "restart_t0": 50, "restart_tmult": 2,
            "backbone": "resnet34",
        },
    },
]

# Tier-3: Resolution + GeM + Dropout experiments (stacking on best Tier-1: LS + SWA)
BATCH_EXPERIMENTS_T3 = [
    {
        "name": "t3_res112",
        "desc": "T3-Exp 1/3: Reduced resolution 112x112 + LS + SWA",
        "overrides": {
            "num_epoch": 250, "arcface_m": 0.35,
            "label_smoothing": 0.1, "use_swa": True,
            "input_size": 112,
        },
    },
    {
        "name": "t3_res112_gem",
        "desc": "T3-Exp 2/3: + GeM pooling",
        "overrides": {
            "num_epoch": 250, "arcface_m": 0.35,
            "label_smoothing": 0.1, "use_swa": True,
            "input_size": 112, "use_gem": True,
        },
    },
    {
        "name": "t3_res112_gem_drop",
        "desc": "T3-Exp 3/3: + Embedding Dropout 0.1",
        "overrides": {
            "num_epoch": 250, "arcface_m": 0.35,
            "label_smoothing": 0.1, "use_swa": True,
            "input_size": 112, "use_gem": True, "embed_dropout": 0.1,
        },
    },
]

# =============================================================================
# HELPERS
# =============================================================================
def print_header(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def ensure_exists(path, label: str) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n  {path}")
    print(f"  [OK] {label}: {path}")


def run_cmd(cmd, cwd=None, desc="Running command") -> None:
    print_header(desc)
    print("Command:\n  " + " ".join(map(str, cmd)) + "\n")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise RuntimeError(f"{desc} failed with exit code {result.returncode}")


def install_requirements() -> None:
    req_file = REPO_PATH / "requirements.txt"
    if req_file.exists():
        # Non-fatal: requirements.txt may have strict version pins that
        # don't match Trinity's Python version. Most deps are pre-installed.
        print_header("Installing dependencies from requirements.txt (best-effort)")
        print(f"  File: {req_file}")
        result = subprocess.run(
            [PYTHON_EXEC, "-m", "pip", "install", "--no-deps", "-r", str(req_file)],
            cwd=str(REPO_PATH),
        )
        if result.returncode != 0:
            print("  [WARN] requirements.txt install had errors — continuing anyway.")
            print("         Trinity likely has the core packages pre-installed.")

    # These are the packages we actually need — best-effort too since
    # Trinity's network may block pip entirely:
    print_header("Installing extra required packages (best-effort)")
    extras = ["matplotlib", "seaborn", "tqdm", "scikit-learn", "joblib", "scipy"]
    print(f"  Packages: {', '.join(extras)}")
    result = subprocess.run(
        [PYTHON_EXEC, "-m", "pip", "install"] + extras,
        cwd=str(REPO_PATH),
    )
    if result.returncode != 0:
        print("  [WARN] Extra packages install failed (network may be blocked).")
        print("         If these are already installed, training will still work.")


def find_latest_log_dir() -> Path:
    log_dirs = [Path(p) for p in glob.glob(str(REPO_PATH / "logs" / "*"))
                if Path(p).is_dir()]
    if not log_dirs:
        raise FileNotFoundError("No logs/* directory found after training.")
    return max(log_dirs, key=lambda p: p.stat().st_mtime)


def find_best_weights() -> Path:
    latest_dir = find_latest_log_dir()
    best_path = latest_dir / "best.pth"
    if not best_path.exists():
        raise FileNotFoundError(f"best.pth not found in:\n  {latest_dir}")
    return best_path


def copy_checkpoints(save_dir: Path, latest_dir: Path) -> Path:
    save_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["best.pth", "last.pth"]:
        src = latest_dir / fname
        if src.exists():
            dst = save_dir / fname
            shutil.copy2(src, dst)
            print(f"  Saved checkpoint: {dst}")
    weights_path = save_dir / "best.pth"
    if not weights_path.exists():
        raise FileNotFoundError(f"best.pth not found at:\n  {weights_path}")
    return weights_path


# =============================================================================
# TRAINING
# =============================================================================
def run_training(args) -> None:
    train_cmd = [
        PYTHON_EXEC, "train.py",
        "--model",       args.model,
        "--train_data",  args.train_data,
        "--test_data",   args.test_data,
        "--batch_size",  str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--num_epoch",   str(args.num_epoch),
        "--seed",        str(args.seed),
        "--eta_min",     str(args.eta_min),
        "--arcface_m",   str(args.arcface_m),
        "--arcface_s",   str(args.arcface_s),
        "--arcface_k",   str(args.arcface_k),
        "--backbone",    args.backbone,
        "--input_size",  str(args.input_size),
        "--embed_dropout", str(args.embed_dropout),
        "--label_smoothing", str(args.label_smoothing),
    ]
    # Tier-3 flags
    if getattr(args, 'use_gem', False):
        train_cmd.append("--use_gem")
    # Tier-1 flags (only add if enabled — they are store_true in train.py)
    if getattr(args, 'use_swa', False):
        train_cmd.append("--use_swa")
        train_cmd.extend(["--swa_start_frac", str(args.swa_start_frac)])
    if getattr(args, 'use_cutmix', False):
        train_cmd.append("--use_cutmix")
    if getattr(args, 'use_warm_restarts', False):
        train_cmd.append("--use_warm_restarts")
        train_cmd.extend(["--restart_t0", str(args.restart_t0)])
        train_cmd.extend(["--restart_tmult", str(args.restart_tmult)])
    run_cmd(train_cmd, cwd=str(REPO_PATH), desc="STEP 4: TRAINING")


# =============================================================================
# EXTERNAL TEST SCRIPTS
# =============================================================================
def run_recognition_test(args, weights_path: Path) -> None:
    cmd = [
        PYTHON_EXEC, "test_face_recognition.py",
        "--model",       args.model,
        "--test_data",   args.test_data,
        "--weights",     str(weights_path),
        "--batch_size",  str(args.test_batch_size),
        "--num_workers", str(args.num_workers),
        "--arcface_k",   str(args.arcface_k),
        "--backbone",    args.backbone,
        "--input_size",  str(args.input_size),
        "--embed_dropout", str(args.embed_dropout),
    ]
    if getattr(args, 'use_gem', False):
        cmd.append("--use_gem")
    if args.no_tta:
        cmd.append("--no_tta")
    run_cmd(cmd, cwd=str(REPO_PATH), desc="STEP 6a: Face Recognition Test")


def run_verification_test(args, weights_path: Path, out_file: Path) -> None:
    pairs_file = REPO_PATH / "data" / "verification_pairs.txt"
    ensure_exists(pairs_file, "Verification pairs file")

    cmd = [
        PYTHON_EXEC, "test_face_verification.py",
        "--model",     args.model,
        "--test_data", args.test_data,
        "--pairs",     str(pairs_file),
        "--weights",   str(weights_path),
        "--out_file",  str(out_file),
    ]
    if args.no_tta:
        cmd.append("--no_tta")
    run_cmd(cmd, cwd=str(REPO_PATH), desc="STEP 6b: Face Verification Test")


# =============================================================================
# INTERNAL DASHBOARD
# =============================================================================
def run_dashboard(args, save_dir: Path, weights_path: Path,
                  verification_json: Path) -> None:
    print_header("STEP 7: INTERNAL METRICS + DASHBOARD")

    # Ensure project modules are importable
    if str(REPO_PATH) not in sys.path:
        sys.path.insert(0, str(REPO_PATH))

    import torch

    try:
        from sklearn.metrics import confusion_matrix, roc_curve, auc
    except ImportError:
        print("  [ERROR] scikit-learn not installed. Cannot compute metrics.")
        print("         Install with: pip install scikit-learn")
        return

    from models.transformer_model import HybridResNetTransformer
    from models.arcface import ArcMarginProduct
    from my_data_class import Lensless_DCT_offline

    # Plotting is optional — Trinity may not have matplotlib
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import seaborn as sns
        HAS_PLOTTING = True
    except ImportError:
        HAS_PLOTTING = False
        print("  [WARN] matplotlib/seaborn not installed. Metrics will be saved but plots skipped.")
        print("         Download metrics.json + results.json and plot locally.")



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model with the SAME ArcFace params it was trained with ──────────
    net = HybridResNetTransformer(
        in_channels=15, embed_dim=512, depth=4, num_heads=8, out_dim=768
    )
    metric_fc = ArcMarginProduct(
        768, 87,
        s=args.arcface_s,    # must match training config
        m=args.arcface_m,
        K=args.arcface_k,
    )

    ckpt = torch.load(weights_path, map_location=device)
    net.load_state_dict(ckpt["net"])
    metric_fc.load_state_dict(ckpt["metric_fc"])
    net.to(device).eval()
    metric_fc.to(device).eval()
    print("  Model loaded successfully.")

    # ── Inference ─────────────────────────────────────────────────────────────
    dataset = Lensless_DCT_offline(args.test_data)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.test_batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    all_preds, all_targets, all_probs = [], [], []
    use_tta = not args.no_tta

    with torch.no_grad():
        for i, (data, targets) in enumerate(loader):
            x = torch.cat([d.to(device) for d in data], dim=1)
            features = net(x, tta=use_tta)
            logits   = metric_fc(features)
            probs    = torch.softmax(logits, dim=1)

            all_preds.append(logits.argmax(1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    print(f"  Evaluated {len(loader)} batches ({len(loader) * args.test_batch_size} samples).")
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_probs   = np.concatenate(all_probs)

    # ── Metrics ───────────────────────────────────────────────────────────────
    acc  = (all_preds == all_targets).mean() * 100
    top5 = np.mean([t in np.argsort(p)[-5:]
                    for t, p in zip(all_targets, all_probs)]) * 100

    conf_scores    = all_probs.max(axis=1)
    binary_correct = (all_preds == all_targets).astype(int)
    fpr_cls, tpr_cls, thresh_cls = roc_curve(binary_correct, conf_scores)
    roc_auc_cls     = auc(fpr_cls, tpr_cls)
    opt_idx_cls     = (tpr_cls - fpr_cls).argmax()
    opt_thresh_cls  = thresh_cls[opt_idx_cls]

    class_acc = [
        (all_preds[all_targets == c] == c).mean() * 100
        if (all_targets == c).sum() > 0 else 0.0
        for c in range(87)
    ]

    print(f"\n  Overall Accuracy : {acc:.2f}%")
    print(f"  Top-5 Accuracy   : {top5:.2f}%")
    print(f"  ROC AUC (proxy)  : {roc_auc_cls:.4f}")
    print(f"  Optimal Threshold: {opt_thresh_cls:.4f}")

    metrics = {
        "recognition_acc": round(float(acc), 4),
        "top5_acc":        round(float(top5), 4),
        "roc_auc_proxy":   round(float(roc_auc_cls), 4),
        "per_class_acc":   [round(float(x), 4) for x in class_acc],
    }

    # ── Verification ROC (from external script results.json) ──────────────────
    fpr_v = tpr_v = roc_auc_v = opt_thresh_v = opt_idx_v = None
    if verification_json.exists():
        try:
            with open(verification_json, "r") as f:
                results = json.load(f)
            if "true_labels" in results and "pred_scores" in results:
                fpr_v, tpr_v, thresh_v = roc_curve(
                    results["true_labels"], results["pred_scores"])
                roc_auc_v   = auc(fpr_v, tpr_v)
                opt_idx_v   = (tpr_v - fpr_v).argmax()
                opt_thresh_v = thresh_v[opt_idx_v]
                metrics["verification_auc"]       = round(float(roc_auc_v), 4)
                metrics["verification_threshold"] = round(float(opt_thresh_v), 4)
                print(f"  Verification AUC : {roc_auc_v:.4f}")
        except Exception as e:
            print(f"  [WARN] Could not parse verification JSON: {e}")

    metrics_path = save_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved: {metrics_path}")

    # ── Plots (only if matplotlib is available) ─────────────────────────────
    if HAS_PLOTTING:
        # Standalone ROC curve
        if fpr_v is not None:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr_v, tpr_v, color="#FF6B6B", lw=2.5,
                     label=f"Verification ROC (AUC = {roc_auc_v:.4f})")
            plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
                     label="Random (AUC = 0.5)")
            plt.scatter(fpr_v[opt_idx_v], tpr_v[opt_idx_v], color="green", s=100,
                        zorder=5, label=f"Optimal Threshold = {opt_thresh_v:.4f}")
            plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate", fontsize=13)
            plt.ylabel("True Positive Rate",  fontsize=13)
            plt.title("ROC Curve — Face Verification", fontsize=15, fontweight="bold")
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(alpha=0.3); plt.tight_layout()
            roc_path = save_dir / "roc_curve.png"
            plt.savefig(roc_path, dpi=150)
            plt.close()
            print(f"  Saved: {roc_path}")

        # Dashboard (4-panel)
        fig = plt.figure(figsize=(20, 16))
        fig.suptitle(f"Evaluation Dashboard — {args.save_name}", fontsize=16,
                     fontweight="bold")
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, 0])
        mask = all_targets < 20
        cm = confusion_matrix(all_targets[mask], all_preds[mask],
                              labels=list(range(20)))
        sns.heatmap(cm, ax=ax1, cmap="Blues", annot=False, linewidths=0.3, cbar=True)
        ax1.set_title("Confusion Matrix (First 20 Classes)", fontweight="bold")
        ax1.set_xlabel("Predicted"); ax1.set_ylabel("True")

        ax2 = fig.add_subplot(gs[0, 1])
        colors = ["#2ecc71" if a >= 80 else "#e67e22" if a >= 50 else "#e74c3c"
                  for a in class_acc]
        ax2.bar(range(87), class_acc, color=colors, width=0.8)
        ax2.axhline(acc, linestyle="--", lw=1.5, label=f"Mean {acc:.1f}%")
        ax2.set_title("Per-Class Accuracy", fontweight="bold")
        ax2.set_xlabel("Class ID"); ax2.set_ylabel("Accuracy (%)")
        ax2.legend(); ax2.set_ylim(0, 105)

        ax3 = fig.add_subplot(gs[1, 0])
        if fpr_v is not None:
            ax3.plot(fpr_v, tpr_v, lw=2.5,
                     label=f"Verification ROC (AUC = {roc_auc_v:.4f})")
            ax3.plot([0, 1], [0, 1], lw=1, linestyle="--")
            ax3.scatter(fpr_v[opt_idx_v], tpr_v[opt_idx_v], s=100, zorder=5,
                        label=f"Threshold = {opt_thresh_v:.4f}")
            ax3.set_title("ROC Curve (Verification)", fontweight="bold")
        else:
            ax3.plot(fpr_cls, tpr_cls, lw=2.5,
                     label=f"Proxy ROC (AUC = {roc_auc_cls:.4f})")
            ax3.plot([0, 1], [0, 1], lw=1, linestyle="--")
            ax3.scatter(fpr_cls[opt_idx_cls], tpr_cls[opt_idx_cls], s=100, zorder=5,
                        label=f"Threshold = {opt_thresh_cls:.4f}")
            ax3.set_title("ROC Curve (Recognition Proxy)", fontweight="bold")
        ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR")
        ax3.legend(); ax3.grid(alpha=0.3)

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        summary = (
            f"{'Metric':<26}{'Value':>12}\n{'─'*40}\n"
            f"{'Recognition Acc':<26}{acc:>11.2f}%\n"
            f"{'Top-5 Accuracy':<26}{top5:>11.2f}%\n"
            f"{'ROC AUC (proxy)':<26}{roc_auc_cls:>12.4f}\n"
            f"{'Test Samples':<26}{len(all_targets):>12}\n"
            f"{'Classes >=80% acc':<26}{sum(a>=80 for a in class_acc):>12}\n"
            f"{'Classes <50% acc':<26}{sum(a<50  for a in class_acc):>12}\n"
        )
        if roc_auc_v is not None:
            summary += f"{'Verification AUC':<26}{roc_auc_v:>12.4f}\n"
        ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=12,
                 verticalalignment="top", fontfamily="monospace",
                 bbox=dict(boxstyle="round", facecolor="#f0f4ff", alpha=0.8))
        ax4.set_title("Summary", fontweight="bold")

        dashboard_path = save_dir / "dashboard.png"
        plt.savefig(dashboard_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Dashboard saved: {dashboard_path}")
    else:
        print("  [SKIP] Plots not generated (matplotlib not available).")
        print("         Download saved_models/<name>/metrics.json and results.json to plot locally.")


    # ── Config file ───────────────────────────────────────────────────────────
    config_path = save_dir / "config.txt"
    with open(config_path, "w") as f:
        f.write(
            f"Experiment : {args.save_name}\n"
            f"Epochs     : {args.num_epoch}  Batch: {args.batch_size}  "
            f"Seed: {args.seed}\n"
            f"ArcFace    : m={args.arcface_m}, s={args.arcface_s}, "
            f"K={args.arcface_k}\n"
            f"Recognition Accuracy : {acc:.2f}%\n"
            f"Top-5 Accuracy       : {top5:.2f}%\n"
        )
        if roc_auc_v is not None:
            f.write(f"Verification AUC     : {roc_auc_v:.4f}\n")
    print(f"  Config saved: {config_path}")
    print(f"\n  All results in: {save_dir}/")
    print(f"  Contents: {os.listdir(save_dir)}")


# =============================================================================
# MAIN
# =============================================================================
def run_single_experiment(args) -> None:
    """Run one complete experiment: train → test → dashboard."""
    # ── Step 4: Training ─────────────────────────────────────────────────────
    if not args.skip_training:
        run_training(args)
    else:
        print_header("STEP 4: SKIPPED (--skip_training)")

    # ── Step 5: Find & copy checkpoints ──────────────────────────────────────
    print_header("STEP 5: FINDING BEST CHECKPOINT")
    latest_dir = find_latest_log_dir()
    best_wt    = find_best_weights()
    print(f"  Latest log dir : {latest_dir}")
    print(f"  Best weights   : {best_wt}")

    save_dir   = REPO_PATH / "saved_models" / args.save_name
    copied_wt  = copy_checkpoints(save_dir, latest_dir)

    # ── Step 6: External test scripts ────────────────────────────────────────
    print_header("STEP 6: EXTERNAL TEST SCRIPTS")
    run_recognition_test(args, copied_wt)

    verification_json = save_dir / "results.json"
    run_verification_test(args, copied_wt, verification_json)

    # ── Step 7: Dashboard ────────────────────────────────────────────────────
    run_dashboard(args, save_dir, copied_wt, verification_json)

    print_header(f"EXPERIMENT '{args.save_name}' COMPLETE")
    print(f"  Results: {save_dir}/")


def main():
    import copy
    parser = build_parser()
    args = parser.parse_args()

    # ── Step 1: Verify paths ─────────────────────────────────────────────────
    print_header("STEP 1: CHECKING PATHS")
    ensure_exists(REPO_PATH, "Repository path")
    ensure_exists(args.train_data, "Training data")
    ensure_exists(args.test_data, "Test data")
    ensure_exists(REPO_PATH / "train.py", "train.py")
    ensure_exists(REPO_PATH / "test_face_recognition.py", "test_face_recognition.py")
    ensure_exists(REPO_PATH / "test_face_verification.py", "test_face_verification.py")

    # ── Step 2: cd into repo ─────────────────────────────────────────────────
    print_header("STEP 2: CHANGING TO REPO DIRECTORY")
    os.chdir(REPO_PATH)
    print(f"  cwd = {os.getcwd()}")
    os.environ["TRAIN_DATA"] = args.train_data
    os.environ["TEST_DATA"]  = args.test_data

    # ── Step 3: Dependencies ──────────────────────────────────────────────
    if args.do_install:
        print_header("STEP 3: INSTALLING DEPENDENCIES")
        install_requirements()
    else:
        print_header("STEP 3: SKIPPED (deps pre-installed, use --do_install to force)")

    # ── Batch mode: run all experiments ───────────────────────────────────────
    if args.batch_experiments or args.batch_tier1 or args.batch_tier2 or args.batch_tier3:
        if args.batch_tier3:
            experiments = BATCH_EXPERIMENTS_T3
        elif args.batch_tier2:
            experiments = BATCH_EXPERIMENTS_T2
        elif args.batch_tier1:
            experiments = BATCH_EXPERIMENTS_T1
        else:
            experiments = BATCH_EXPERIMENTS
        print_header("BATCH MODE: Running {} experiments sequentially".format(
            len(experiments)))
        for i, exp in enumerate(experiments):
            print(f"\n{'#'*90}")
            print(f"# {exp['desc']}")
            print(f"# Save name: {exp['name']}")
            print(f"# Overrides: {exp['overrides']}")
            print(f"{'#'*90}")

            # Create a copy of args with experiment-specific overrides
            exp_args = copy.deepcopy(args)
            exp_args.save_name = exp["name"]
            for key, val in exp["overrides"].items():
                setattr(exp_args, key, val)

            try:
                run_single_experiment(exp_args)
            except Exception as e:
                print(f"\n  [ERROR] Experiment '{exp['name']}' FAILED: {e}")
                print(f"  Continuing to next experiment...\n")
                continue

        print_header("ALL BATCH EXPERIMENTS DONE!")
        print("  Results are in saved_models/<experiment_name>/")
        for exp in experiments:
            sd = REPO_PATH / "saved_models" / exp["name"]
            status = "✓" if sd.exists() else "✗ (failed)"
            print(f"    {status}  {exp['name']}")
    else:
        # ── Single experiment mode ────────────────────────────────────────────
        run_single_experiment(args)
        print_header("DONE!")
        print("  Training, testing, and dashboard generation completed.")
        print(f"  Results: {REPO_PATH / 'saved_models' / args.save_name}/")


if __name__ == "__main__":
    main()
