"""Recovered rejected prototype: independent branches still compete through argmax.

Its calibration failed repair visibility. Use modular_net.py for the routed design.
"""
import numpy as np, torch
from sklearn.datasets import load_digits
from torch import nn

H1, H2 = 16, 8


class Branch(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1, self.l2, self.l3 = nn.Linear(64, H1), nn.Linear(H1, H2), nn.Linear(H2, 1)

    def forward(self, x):
        return self.l3(torch.relu(self.l2(torch.relu(self.l1(x)))))


class BranchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList(Branch() for _ in range(10))

    def forward(self, x):
        return torch.cat([b(x) for b in self.branches], 1)


def data():
    x, y = load_digits(return_X_y=True)
    x = (x / 16).astype(np.float32); y = y.astype(np.int64)
    o = np.random.default_rng(21).permutation(len(x))
    x, y = x[o], y[o]
    return x, y, slice(0, 900), slice(900, 1300), slice(1300, None)


def trained():
    x, y, tr, pr, ev = data()
    torch.manual_seed(21)
    net = BranchNet()
    opt = torch.optim.Adam(net.parameters(), 5e-3)
    for _ in range(600):
        opt.zero_grad(); torch.nn.functional.cross_entropy(net(torch.tensor(x[tr])), torch.tensor(y[tr])).backward(); opt.step()
    return net


def mirror(b):
    b.l1.weight.copy_(b.l1.weight.reshape(-1, 8, 8).flip(2).reshape(-1, 64))


def shift(b):
    b.l1.weight.copy_(b.l1.weight.reshape(-1, 8, 8).roll(2, 1).reshape(-1, 64))


def flip_units(b):
    idx = list(range(0, H2, 2)); b.l2.weight[idx] *= -1; b.l2.bias[idx] *= -1


def rewire(b):
    p = np.random.default_rng(5).permutation(H1); b.l2.weight.copy_(b.l2.weight[:, p])


def shift_bias(b):
    b.l2.bias.add_(torch.tensor([3.0, -3.0] * (H2 // 2)))


def flip_output(b):
    b.l3.weight.mul_(-1); b.l3.bias.mul_(-1)


def weaken(b):
    b.l3.weight.mul_(0.1)


# digit: faults. Digit 7 has two that must be repaired together; digits 8 and 9 are healthy.
FAULTS = {0: [flip_output], 1: [shift_bias], 2: [weaken], 3: [flip_units], 4: [rewire], 5: [mirror], 6: [shift], 7: [rewire, flip_units]}


def broken(only=None, skip=None):
    net = trained()
    with torch.no_grad():
        for digit, faults in FAULTS.items():
            if (only is None or digit in only) and (skip is None or digit not in skip):
                for f in faults:
                    f(net.branches[digit])
    return net


def accuracy(net):
    x, y, tr, pr, ev = data()
    with torch.no_grad():
        return float((net(torch.tensor(x[ev])).argmax(1).numpy() == y[ev]).mean())
