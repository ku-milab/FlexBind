import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc

TASK_NAMES = ['disorder', 'protein', 'rna', 'dna']
BINDING_IDX = [1, 2, 3]  # macro/micro 집계 대상 (disorder 제외)
NAN = float('nan')


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def masked_bce_with_logits(logits, targets, mask, pos_weight=None):
    return F.binary_cross_entropy_with_logits(logits[mask], targets[mask], pos_weight=pos_weight)


def focal_bce_with_logits(logits, targets, mask, gamma=2.0, alpha=0.75):
    logits, targets = logits[mask], targets[mask]
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    p = torch.sigmoid(logits)
    pt = targets * p + (1 - targets) * (1 - p)
    alpha_t = targets * alpha + (1 - targets) * (1 - alpha)
    return (alpha_t * (1 - pt) ** gamma * bce).mean()


def dice_loss(logits, targets, mask, eps=1e-6):
    probs, targets = torch.sigmoid(logits)[mask], targets[mask]
    inter = (probs * targets).sum()
    return 1 - (2 * inter + eps) / (probs.sum() + targets.sum() + eps)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _best_f1(P, R, TH):
    """PR 곡선에서 F1이 최대인 지점의 (f1, threshold, precision, recall)."""
    # P[i], R[i]가 TH[i]에 대응. 마지막 점 (P=1, R=0)은 threshold가 없으므로 제외.
    f1 = 2 * P[:-1] * R[:-1] / (P[:-1] + R[:-1] + 1e-12)
    i = int(np.argmax(f1))
    return float(f1[i]), float(TH[i]), float(P[i]), float(R[i])


def _has_both_classes(y_true):
    return len(y_true) > 0 and y_true.min() != y_true.max()


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #
def plot_roc_pr_curves(metrics, out_dir, suffix=""):
    """태스크별 PR 곡선을 저장하고 best F1 리스트를 반환."""
    best_f1_per_task = []

    for t, name in enumerate(TASK_NAMES):
        y_true = metrics['roc_raw']['y_true'][t]
        y_prob = metrics['roc_raw']['y_prob'][t]

        if not _has_both_classes(y_true):
            print(f"[WARN] Cannot plot PR for task {name} (no positive or negative examples)")
            best_f1_per_task.append(NAN)
            continue

        P, R, TH = precision_recall_curve(y_true, y_prob)
        best_f1, best_th, best_p, best_r = _best_f1(P, R, TH)

        plt.figure(figsize=(5, 5))
        plt.plot(R, P, label=f"AUC={metrics['auc'][t]:.4f}, AUPR={auc(R, P):.4f}, "
                             f"F1={best_f1:.4f}, Rec={best_r:.4f}")
        plt.scatter(best_r, best_p, color='red', s=40, label=f"Best-F1@th={best_th:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve - {name}")
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/pr_{name}{suffix}.png", dpi=200)
        plt.close()

        print(f"[PLOT] Saved PR curve for '{name}{suffix}' (Best F1={best_f1:.4f})")
        best_f1_per_task.append(best_f1)

    return best_f1_per_task


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    n_tasks = len(TASK_NAMES)

    total_loss, nsteps = 0.0, 0
    tp, tn, fp, fn = (torch.zeros(n_tasks, device=cfg.device) for _ in range(4))
    y_true_all = [[] for _ in range(n_tasks)]
    y_prob_all = [[] for _ in range(n_tasks)]

    for batch in loader:
        H = batch['emb'].to(cfg.device)
        Y = batch['label'].to(cfg.device)
        M = batch['mask'].to(cfg.device)
        logits = model(H, M)

        loss_sum = 0.0
        for j, name in enumerate(TASK_NAMES):
            y = Y[:, :, j]
            valid = M & (y > -50)  # 패딩 및 라벨 없는 위치 제외

            loss = masked_bce_with_logits(logits[name], y, valid)
            if cfg.use_focal:
                loss = loss + focal_bce_with_logits(logits[name], y, valid, alpha=cfg.alphas[name])
            if cfg.use_dice:
                loss = loss + dice_loss(logits[name], y, valid)
            loss_sum = loss_sum + loss

            if valid.any():
                prob = torch.sigmoid(logits[name])[valid]
                pred = (prob >= cfg.thresholds.get(name, 0.5)).float()
                tgt = y[valid].float()

                tp[j] += (pred * tgt).sum()
                fp[j] += (pred * (1 - tgt)).sum()
                fn[j] += ((1 - pred) * tgt).sum()
                tn[j] += ((1 - pred) * (1 - tgt)).sum()

                y_true_all[j].append(tgt.cpu().numpy())
                y_prob_all[j].append(prob.cpu().numpy())

        total_loss += loss_sum.item()
        nsteps += 1

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    y_true_cat = [np.concatenate(v) if v else np.array([]) for v in y_true_all]
    y_prob_cat = [np.concatenate(v) if v else np.array([]) for v in y_prob_all]

    auc_list, thr_youden, thr_f1 = [], [], []
    for y_true, y_prob in zip(y_true_cat, y_prob_cat):
        if not _has_both_classes(y_true):
            auc_list.append(NAN)
            thr_youden.append(NAN)
            thr_f1.append(NAN)
            continue

        auc_list.append(float(roc_auc_score(y_true, y_prob)))

        fpr, tpr, thr = roc_curve(y_true, y_prob)
        thr_youden.append(float(thr[np.argmax(tpr - fpr)]))

        thr_f1.append(_best_f1(*precision_recall_curve(y_true, y_prob))[1])

    TP, FP, FN = (float(x[BINDING_IDX].sum()) for x in (tp, fp, fn))

    return {
        'loss': total_loss / max(nsteps, 1),
        'precision': precision.tolist(),
        'recall': recall.tolist(),
        'f1': f1.tolist(),
        'auc': auc_list,
        'macro': {
            'sensitivity': float(recall[BINDING_IDX].mean()),
            'f1': float(f1[BINDING_IDX].mean()),
        },
        'micro': {
            'sensitivity': TP / (TP + FN + 1e-8),
            'f1': 2 * TP / (2 * TP + FP + FN + 1e-8),
        },
        'best_thr': thr_youden,
        'best_thr_f1': thr_f1,
        'confusion': {
            'tp': tp.tolist(),
            'fp': fp.tolist(),
            'fn': fn.tolist(),
            'tn': tn.tolist(),
        },
        'actual': {
            'pos': (tp + fn).tolist(),
            'neg': (tn + fp).tolist(),
        },
        'roc_raw': {'y_true': y_true_cat, 'y_prob': y_prob_cat},
    }
