"""CPU-only benchmark construction. Training creates the fixture, never repairs it.

Ten binary digit detectors, with an explicit one-hot query selecting one branch.
This avoids argmax competition between damaged branches. Split images BEFORE
making queries; reserve a calibration split distinct from practice and evaluation.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(64, 16)
        self.l2 = nn.Linear(16, 8)
        self.l3 = nn.Linear(8, 2)

    def forward(self, x):
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))


class RoutedDigits(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList(Branch() for _ in range(10))

    def branch_logits(self, x):
        logits = torch.stack([branch(x) for branch in self.branches], dim=1)
        return logits[:, :, 1] - logits[:, :, 0]

    def forward(self, x):
        value = (self.branch_logits(x[:, :64]) * x[:, 64:]).sum(1)
        return torch.stack([torch.zeros_like(value), value], dim=1)


def build():
    return RoutedDigits()


def image_splits(seed=21):
    x, y = load_digits(return_X_y=True)
    x, y = (x / 16).astype(np.float32), y.astype(np.int64)
    train, rest = train_test_split(np.arange(len(x)), train_size=900, stratify=y, random_state=seed)
    calibration, rest = train_test_split(rest, train_size=250, stratify=y[rest], random_state=seed)
    practice, evaluation = train_test_split(rest, train_size=300, stratify=y[rest], random_state=seed)
    return {name: (x[idx], y[idx]) for name, idx in
            zip(("train", "calibration", "practice", "evaluation"), (train, calibration, practice, evaluation))}


def queries(images, labels, seed=21):
    """Equal positives/negatives and counts per branch, one query per source image.

    No repeated image can leak across the search's practice/selection split or
    inflate its sample count merely by being queried for a different digit.
    """
    rng = np.random.default_rng(seed)
    count = min(int((labels == digit).sum()) for digit in range(10)) // 2
    if count < 1:
        raise ValueError("need at least two images per digit")
    positives = [rng.choice(np.flatnonzero(labels == digit), count, replace=False) for digit in range(10)]
    available = np.ones(len(labels), dtype=bool)
    available[np.concatenate(positives)] = False
    inputs, targets = [], []
    for digit in range(10):
        pos = positives[digit]
        neg = rng.choice(np.flatnonzero(available & (labels != digit)), count, replace=False)
        available[neg] = False
        idx = np.concatenate([pos, neg])
        route = np.zeros((len(idx), 10), dtype=np.float32)
        route[:, digit] = 1
        inputs.append(np.concatenate([images[idx], route], axis=1))
        targets.append((labels[idx] == digit).astype(np.int64))
    return np.concatenate(inputs), np.concatenate(targets)


def trained(seed=21):
    torch.manual_seed(seed)
    net = build()
    images, labels = image_splits(seed)["train"]
    x = torch.from_numpy(images)
    target = torch.tensor(labels[:, None] == np.arange(10), dtype=torch.float32)
    positive_weight = (1 - target.mean(0)) / target.mean(0)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.005)
    for _ in range(600):
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            net.branch_logits(x), target, pos_weight=positive_weight)
        loss.backward()
        optimizer.step()
    return net.eval()


FAULTS = {0: "output swap", 1: "hidden bias suppression", 2: "output offset",
          3: "hidden sign flips", 4: "rewiring", 5: "input mirror", 6: "input shift",
          7: "rewiring + hidden sign flips"}


def damage(clean):
    net = deepcopy(clean)
    with torch.no_grad():
        for digit in FAULTS:
            b = net.branches[digit]
            if digit == 0:
                b.l3.weight.copy_(b.l3.weight.flip(0))
                b.l3.bias.copy_(b.l3.bias.flip(0))
            elif digit == 1:
                b.l2.bias.sub_(12.)
            elif digit == 2:
                b.l3.bias[1].add_(20.)
            elif digit in (5, 6):
                w = b.l1.weight.reshape(-1, 8, 8)
                b.l1.weight.copy_((w.flip(2) if digit == 5 else w.roll(2, 1)).reshape(-1, 64))
            if digit in (4, 7):
                permutation = np.random.default_rng(5).permutation(16)
                b.l2.weight.copy_(b.l2.weight[:, permutation])
            if digit in (3, 7):
                # Corrupt the seven strongest output contributors, leaving one
                # untouched so the inverse is not forbidden whole-tensor scaling.
                indices = torch.topk((b.l3.weight[1] - b.l3.weight[0]).abs(), 7).indices
                b.l2.weight[indices] *= -1
                b.l2.bias[indices] *= -1
    return net


def restore(broken, clean, digits):
    net = deepcopy(broken)
    for digit in digits:
        net.branches[digit].load_state_dict(clean.branches[digit].state_dict())
    return net


def scores(net, inputs, targets):
    with torch.no_grad():
        prediction = net(torch.from_numpy(inputs)).argmax(1).numpy()
    correct = prediction == targets
    routes = inputs[:, 64:].argmax(1)
    return float(correct.mean()), [float(correct[routes == digit].mean()) for digit in range(10)]
