"""A trained network with faults that do not show up as one digit going wrong."""
import numpy as np, torch
from sklearn.datasets import load_digits
from wsplab.runners.torch import mlp

SIZES = [64, 48, 48, 10]


def data():
    x, y = load_digits(return_X_y=True)
    x = (x / 16).astype(np.float32); y = y.astype(np.int64)
    o = np.random.default_rng(11).permutation(len(x))
    x, y = x[o], y[o]
    return x, y, slice(0, 900), slice(900, 1300), slice(1300, None)


def build(strength: int = 1):
    """Returns (broken net, clean accuracy, broken accuracy, what was done)."""
    x, y, tr, pr, ev = data()
    torch.manual_seed(11)
    net = mlp(SIZES, "relu")
    opt = torch.optim.Adam(net.parameters(), 5e-3)
    for _ in range(500):
        opt.zero_grad(); torch.nn.functional.cross_entropy(net(torch.tensor(x[tr])), torch.tensor(y[tr])).backward(); opt.step()
    acc = lambda: float((net(torch.tensor(x[ev])).argmax(1).numpy() == y[ev]).mean())
    clean = acc()
    rng = np.random.default_rng(12)
    u = rng.permutation(48)
    v = rng.permutation(48)
    with torch.no_grad():
        w1, w2, b2 = net.layers[0].weight, net.layers[2].weight, net.layers[2].bias
        mirrored = u[:20].tolist()                       # these units see the image mirrored left to right
        w1[mirrored] = w1[mirrored].reshape(-1, 8, 8).flip(2).reshape(-1, 64)
        wired = u[20:36].tolist()                        # the second layer reads these units in a rotated order
        w2[:, wired] = w2[:, wired[1:] + wired[:1]].clone()
        flipped = v[:14].tolist()                         # these second-layer units have their input negated
        w2[flipped] = -w2[flipped]
        shifted = v[14:26].tolist()                       # and these have a large bias
        b2[shifted] += torch.tensor(rng.choice([-4.0, 4.0], 12), dtype=b2.dtype)
    return net, clean, acc(), {"mirrored": mirrored, "wired": wired, "flipped": flipped, "shifted": shifted}
