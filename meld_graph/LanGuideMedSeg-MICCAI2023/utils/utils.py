from torch.utils.data import Sampler
import numpy as np
import random


class LesionOversampleSampler(Sampler):
    """
    Сэмплер, который берёт ВСЕ healthy-примеры ровно по одному разу,
    а lesion-примеры — с replacement, чтобы заполнить всю эпоху.
    """

    def __init__(self, labels, seed=42):
        self.labels = labels
        random.seed(seed)
        # индексы здоровых и lesion
        self.hc_idx = [i for i, label in enumerate(labels) if label == 0]
        self.les_idx = [i for i, label in enumerate(labels) if label == 1]
        # хотим ровно len(labels) выборок за эпoху
        self.epoch_size = len(labels)

    def __iter__(self):
        # начинаем с всех hc-индексов
        idxs = self.hc_idx.copy()
        # сколько нужно докинуть lesion'ов
        n_les_to_sample = self.epoch_size - len(idxs)
        # добавляем lesion с replacement
        idxs += random.choices(self.les_idx, k=n_les_to_sample)
        # перемешиваем всю эпоху
        random.shuffle(idxs)
        return iter(idxs)

    def __len__(self):
        return self.epoch_size


def summarize_ci(scores, B=10_000, alpha=0.05, seed=42):
    x = np.asarray(scores, dtype=float)
    x = x[~np.isnan(x)]
    N = x.size
    if N == 0:
        return np.nan, np.nan, np.nan
    if N == 1:
        return float(x[0]), float(x[0]), float(x[0])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(B, N))  # 10k resamples
    boot_meds = np.median(x[idx], axis=1)  # median in each resample
    lo, hi = np.percentile(boot_meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return float(np.median(x)), float(lo), float(hi)


def compute_adaptive_threshold(prediction: np.ndarray) -> float:
    mp = prediction.max()
    if mp >= 0.5:
        return 0.5
    return max(mp * 0.2, 0.01)
