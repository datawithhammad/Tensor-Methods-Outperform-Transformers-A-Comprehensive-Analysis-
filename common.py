"""Shared utilities: environment setup, seeding, timing, memory tracking."""
import os

# Anaconda's MKL and pip torch each bundle libiomp5md.dll; allow both.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import random
import time

import numpy as np
import psutil
import torch

SEED = 42
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rss_mb():
    return psutil.Process().memory_info().rss / (1024 * 1024)


class RunTracker:
    """Tracks wall time and peak resident memory over a run."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.peak_mb = rss_mb()

    def sample(self):
        self.peak_mb = max(self.peak_mb, rss_mb())

    def elapsed(self):
        return time.perf_counter() - self.t0
