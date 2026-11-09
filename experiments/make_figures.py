"""Draw the figures in FINDINGS.md from the result files in this folder: python experiments/make_figures.py

Every chart is written twice, for light and dark backgrounds (docs/figures/<name>-light.png and -dark.png). Colour
follows the entity across all charts: grey = the original network, blue = an editor working alone, orange = a council,
aqua = a scripted (non-LLM) baseline. Numbers come from the JSON files under experiments/*/results and the experiment
reports; the few early results that exist only in text are constants below, with their source.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "figures"

THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "original": "#8a8985",
              "solo": "#2a78d6", "council": "#eb6834", "script": "#1baf7a"},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#c3c2b7", "grid": "#33332f", "original": "#8a8985",
             "solo": "#3987e5", "council": "#d95926", "script": "#199e70"},
}


def load(path):
    return json.loads((HERE / path).read_text())


def style(ax, t, title, xlabel=None):
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["muted"], labelsize=9)
    ax.grid(axis="x", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", color=t["text"], fontsize=11, pad=8)
    if xlabel:
        ax.set_xlabel(xlabel, color=t["muted"], fontsize=9)


def figure(t, w, h):
    fig = plt.figure(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(t["surface"])
    return fig


def save(fig, name, mode):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.png", facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def hbars(ax, t, rows, unit_fmt, xmax, ref=None):
    """rows: (label, value, half_ci or None, role). Thin horizontal bars, value labelled at the end."""
    for i, (label, value, ci, role) in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, value, height=0.55, color=t[role], edgecolor=t["surface"], linewidth=2)
        if ci:
            ax.errorbar(value, y, xerr=ci, color=t["text"], linewidth=1, capsize=3)
        ax.text(value + (ci or 0) + xmax * 0.012, y, unit_fmt(value), va="center", color=t["text"], fontsize=9)
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], color=t["text"], fontsize=9)
    ax.set_xlim(0, xmax)
    if ref:
        x, text = ref
        ax.axvline(x, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
        ax.text(x - xmax * 0.01, -0.45, text, color=t["muted"], fontsize=8, ha="right", va="bottom")


# --- 1. Seaquest -----------------------------------------------------------------------------------------------------
def seaquest(mode):
    t = THEMES[mode]
    strong = load("10_seaquest/results/trial1/comparison.json")
    cheap = load("10_seaquest/results/cheap1/comparison.json")
    roles = load("10_seaquest/results/roles1/comparison.json")
    s = {r["session"]: r for r in strong["table"]}
    c = {r["session"]: r for r in cheap["table"]}
    ci = lambda r: 1.96 * r["se"]  # noqa: E731
    rows = [
        ("Original (Astra's network)", strong["original"]["mean_return"], 1.96 * strong["original"]["se"], "original"),
        ("Claude Opus alone", s["independent/anthropic"]["mean_return"], ci(s["independent/anthropic"]), "solo"),
        ("gpt-6.1-terra alone", s["independent/openai"]["mean_return"], ci(s["independent/openai"]), "solo"),
        ("Strong council (peers)", s["peers/openai"]["mean_return"], ci(s["peers/openai"]), "council"),
        ("Strong council with roles", roles["means"]["roles/openai"], None, "council"),
        ("Best cheap model alone (Luna)", c["independent/openai"]["mean_return"], ci(c["independent/openai"]), "solo"),
        ("Cheap council (kept original)", cheap["original"]["mean_return"], None, "council"),
    ]
    fig = figure(t, 7.2, 3.6)
    ax = fig.add_subplot()
    hbars(ax, t, rows, lambda v: f"{v:.1f}", 100, ref=(88.9, "PPO, same network, 500M steps: 88.9"))
    style(ax, t, "MinAtar Seaquest: mean return on 2,000 fresh episodes", "mean return per episode (95% CI)")
    save(fig, "seaquest", mode)


# --- 2. Council minus best solo, every task --------------------------------------------------------------------------
def council_vs_solo(mode):
    t = THEMES[mode]
    digits = load("04_routed_digits/peer-confirmation-results.json")
    cifar = load("05_cifar100/natural-cifar100-results.json")
    sea = load("10_seaquest/results/trial1/comparison.json")
    roles = load("10_seaquest/results/roles1/comparison.json")
    sp = load("09_spurious_imagenet/results/cluster-bootstrap.json")["peers_minus_portfolio_valid"]
    claude_gain_se = next(r for r in sea["table"] if r["session"] == "independent/anthropic")["minus_original"]["standard_error"]
    rows = [  # label, unit, difference, (low, high)
        ("Routed digits (synthetic)", "accuracy, points",
         100 * digits["selected_peer_minus_independent"], 1.96 * 100 * digits["paired_standard_error"]),
        ("Spurious ImageNet, ResNet-50", "balanced score, points", 100 * sp["difference"],
         (100 * sp["ci_low"], 100 * sp["ci_high"])),
        ("CIFAR-100 ResNet-20 (council arm invalid)", "accuracy, points",
         100 * cifar["peers_minus_independent"]["difference"],
         1.96 * 100 * cifar["peers_minus_independent"]["paired_standard_error"]),
        ("Seaquest: strong council", "return", sea["primary_peers_minus_portfolio_valid_only"]["difference"],
         1.96 * sea["primary_peers_minus_portfolio_valid_only"]["standard_error"]),
        ("Seaquest: council with roles", "return", roles["role_pick_minus_claude_solo"]["difference"],
         1.96 * roles["role_pick_minus_claude_solo"]["standard_error"]),
        ("Seaquest: cheap council vs Opus alone", "return",
         load("10_seaquest/results/cheap1/comparison.json")["original"]["mean_return"]
         - next(r for r in sea["table"] if r["session"] == "independent/anthropic")["mean_return"], 1.96 * claude_gain_se),
    ]
    fig = figure(t, 7.2, 0.85 * len(rows) + 1.3)
    axes = fig.subplots(len(rows), 1)
    for ax, (label, unit, d, ci) in zip(axes, rows):
        lo, hi = (d - ci, d + ci) if not isinstance(ci, tuple) else ci
        span = max(abs(lo), abs(hi)) * 1.35 + 0.5
        ax.axvline(0, color=t["muted"], linewidth=1)
        ax.plot([lo, hi], [0, 0], color=t["council"], linewidth=2, solid_capstyle="round")
        ax.plot([d], [0], "o", color=t["council"], markersize=8, markeredgecolor=t["surface"], markeredgewidth=2)
        ax.set_xlim(-span, span)
        ax.set_yticks([])
        ax.set_ylim(-1, 1)
        style(ax, t, None)
        ax.grid(False)
        ax.spines["left"].set_visible(False)
        ax.text(-span, 0.55, label, color=t["text"], fontsize=9.5, va="bottom")
        ax.text(span, 0.55, f"{d:+.2f} {unit}", color=t["muted"], fontsize=8.5, va="bottom", ha="right")
    fig.suptitle("Council's pick minus the best editor alone (95% CI; right of 0 = council better)",
                 x=0.02, ha="left", color=t["text"], fontsize=11)
    fig.text(0.02, -0.01, "Gemma math and FinQA: tie, both arms kept the original model (no candidate beat it).",
             color=t["muted"], fontsize=8.5)
    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    save(fig, "council-vs-solo", mode)


# --- 3. Spurious ImageNet trade-off ----------------------------------------------------------------------------------
def spurious(mode):
    t = THEMES[mode]
    r = load("09_spurious_imagenet/results/comparison.json")
    fig = figure(t, 7.2, 4.2)
    ax = fig.add_subplot()
    bar = r["bar"]
    ax.axvspan(0, 100 * bar["target_spurious_hit"], ymin=0, ymax=1, color=t["grid"], alpha=0.0)
    ax.axvline(100 * bar["target_spurious_hit"], color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
    ax.axhline(100 * bar["target_clean_top1"], color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
    ax.text(100 * bar["target_spurious_hit"] - 0.5, 80.3, "pass: left of and above both lines",
            color=t["muted"], fontsize=8, ha="right")
    pts = [("Original", r["baseline"], "original", True)]
    for row in r["rows"]:
        who = row["editor"].split(":")[1].split("-")[0].replace("gpt", "terra").replace("claude", "Claude").replace("gemini", "Gemini")
        role = "council" if row["arm"] == "peers" else "solo"
        pts.append((("peer " if role == "council" else "") + who + ("" if row["protocol_valid"] else " (invalid)"),
                    row, role, row["protocol_valid"]))
    for k, v in r["recipes"].items():
        pts.append((f"scripted recipe {k}", v, "script", True))
    nudge = {"Claude": (6, 6), "peer Gemini": (-8, 8), "peer terra": (6, -12), "peer Claude": (6, 4)}
    for label, m, role, valid in pts:
        x, y = 100 * m["target_spurious_hit"], 100 * m["target_clean_top1"]
        ax.plot([x], [y], "o", markersize=8, color=t[role] if valid else t["surface"],
                markeredgecolor=t[role] if valid else t["solo"], markeredgewidth=2 if not valid else 1.5)
        dx, dy = nudge.get(label, (6, 4))
        ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points", color=t["text"], fontsize=8,
                    ha="right" if dx < 0 else "left")
    for role, name in (("original", "original"), ("solo", "editor alone"), ("council", "council member"),
                       ("script", "scripted baseline")):
        ax.plot([], [], "o", color=t[role], label=name)
    leg = ax.legend(loc="lower right", frameon=False, fontsize=8.5)
    for txt in leg.get_texts():
        txt.set_color(t["text"])
    ax.set_xlim(5, 50)
    ax.set_ylim(68, 81)
    style(ax, t, "Spurious ImageNet: fewer false calls cost clean accuracy", "target classes: spurious photos called the absent class (%)")
    ax.set_ylabel("target classes: clean top-1 accuracy (%)", color=t["muted"], fontsize=9)
    ax.grid(axis="both", color=t["grid"], linewidth=0.8)
    fig.text(0.02, -0.04, "The third pass criterion, overall clean accuracy within 0.5 points, is not drawn: recipe s2 fails it "
             "(-0.55). Gemini's solo edit broke the no-simulation rule.", color=t["muted"], fontsize=8)
    save(fig, "spurious-imagenet", mode)


# --- 4. Early hard digit task ----------------------------------------------------------------------------------------
def hard_digits(mode):
    """Source: docs (pre-0.1) council.md, 'A harder task': 64-48-48-10 digit classifier with 62 planted faults."""
    t = THEMES[mode]
    arms = [("gpt-6.1-terra alone", [47.9, 47.9, 47.9], "solo"), ("Claude Opus alone", [49.5, 51.1, 46.1], "solo"),
            ("Gemini argon alone", [49.5, 49.5, 47.7], "solo"), ("Chair council", [49.7, 85.7, 47.9], "council"),
            ("Chair council, no self-votes", [47.9, 48.9, 47.9], "council"),
            ("Pre-registered repeat (5 runs)", [47.1, 35.2, 35.2, 35.2, 35.2], "council"),
            ("Cooperative team (5 runs)", [49.7, 54.5, 47.9, 47.9, 47.9], "council")]
    fig = figure(t, 7.2, 3.6)
    ax = fig.add_subplot()
    for i, (label, vals, role) in enumerate(arms):
        y = len(arms) - 1 - i
        ax.plot(vals, [y] * len(vals), "o", color=t[role], markersize=8, markeredgecolor=t["surface"], markeredgewidth=2,
                alpha=0.9)
    ax.axvline(35.2, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
    ax.axvline(94.6, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
    ax.text(35.6, -0.75, "damaged 35.2", color=t["muted"], fontsize=8)
    ax.text(94.2, -0.75, "clean 94.6", color=t["muted"], fontsize=8, ha="right")
    ax.annotate("4 runs, no edit", (35.2, 1), xytext=(8, 8), textcoords="offset points", color=t["text"], fontsize=8)
    ax.annotate("one run: 85.7", (85.7, len(arms) - 4), xytext=(-8, 8), textcoords="offset points", ha="right",
                color=t["text"], fontsize=8)
    ax.set_yticks(range(len(arms)), [a[0] for a in reversed(arms)], color=t["text"], fontsize=9)
    ax.set_xlim(30, 100)
    ax.set_ylim(-1, len(arms) - 0.4)
    style(ax, t, "Early test: a digit classifier with 62 planted faults (each dot is one run)", "test accuracy (%)")
    save(fig, "early-hard-digits", mode)


# --- 5. Gemma ---------------------------------------------------------------------------------------------------------
def gemma(mode):
    t = THEMES[mode]
    s = load("11_gemma_math/results/summary.json")
    names = {"openai": "terra", "anthropic": "Claude", "gemini": "Gemini"}
    base = s["rows"][0]["baseline"]
    rows = [("Original", 100 * base, None, "original")]
    for r in s["rows"]:
        if r.get("changed_entries"):
            rows.append(((("peer " if r["arm"] == "peers" else "") + names[r["provider"]] + (" alone" if r["arm"] != "peers" else "")),
                         100 * r["selection_score"], None, "council" if r["arm"] == "peers" else "solo"))
    fig = figure(t, 7.2, 2.4)
    ax = fig.add_subplot()
    hbars(ax, t, rows, lambda v: f"{v:.1f}%", 14)
    style(ax, t, "Gemma 3 270M math: every submitted edit scored below the original",
          "accuracy on 600 held-out practice problems, 2 samples each (%)")
    fig.text(0.02, -0.16, "Sessions that submitted no edit (Gemini in both arms, the Claude peer) are not shown; all hit their spending caps.",
             color=t["muted"], fontsize=8)
    save(fig, "gemma-math", mode)


if __name__ == "__main__":
    for mode in THEMES:
        for draw in (seaquest, council_vs_solo, spurious, hard_digits, gemma):
            draw(mode)
    print(sorted(p.name for p in OUT.glob("*.png")))
