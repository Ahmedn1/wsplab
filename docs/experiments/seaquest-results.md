# MinAtar Seaquest: council versus solo, replicate 1

Task: improve the Seaquest policy network that GPT-6 Astra wrote, one-shot, in the paper's Experiment 5. Astra's
network scored 53.1; PPO with the same 600-32-32-6 architecture reached 88.9. Approved as stage 1 by the user
("Approve stage 1"). Per-session caps: OpenAI $0.75, Claude $2.50, Gemini $2.00. **Spent $7.78** (peers $3.92,
independent $3.86). Artifacts: `~/.wsplab/seaquest/trial1/` (`comparison.json`, `protocol-audit.json`).

## Setup (fixed before any paid call)

- **Network:** `networks/seaquest/hand.npz` from the paper's release, with tensors W1..b3 in the paper's
  orientation. Reproduced at 52.9 ± 1.4 over 1,000 episodes, against the paper's 53.1 ± 1.4.
- **Scoring (the paper's):** MinAtar 1.0.15, the paper's 6-channel input, actions sampled from the softmax, sticky
  actions 0.1, 10,000-step cap. Evaluation runs 2,000 fresh seeds that no editor saw.
- **Practice:** 2,000 seeds. The editors' work half has 1,000 seeds; the other 1,000 pick each arm's candidate.
- **Rules (the paper's):**
  - Not allowed: training or iterative fitting, and running or simulating the network (or playing the game with
    it) in Python.
  - Allowed: any analysis of weights and forward-tool results, and directly computed edits.
  - Disqualification for simulation, decided in advance.
- **Against overfitting:**
  - Editors fix DEV and HOLDOUT episodes, and every pass plays at least 100 episodes.
  - Edit budget: 1,024 entries (5%), enforced by `save_edit`.
- **Arms:**
  - Peers: OpenAI, Claude and Gemini, sharing every step.
  - Independent: the same three alone. This gives each solo result and the portfolio pick.
- **Budget per editor:** 8 passes, all required, as in the paper.

## Results on 2,000 fresh episodes

| Session | Entries changed | Selection score | Evaluation return | Gain over original | Mean length |
|---|---:|---:|---:|---:|---:|
| Original (Astra's network) | — | 53.95 | **52.13** ± 0.97 | — | 1,388 |
| **Peer OpenAI (peers' pick)** | 32 | 71.99 | **74.80** ± 0.91 | +22.67 ± 1.19 | 1,994 |
| Peer Claude | 31 | 71.50 | 74.66 ± 0.92 | +22.53 ± 1.18 | 1,983 |
| Peer Gemini | 32 | 71.99 | 74.80 ± 0.91 | +22.67 ± 1.19 | 1,994 |
| Solo OpenAI | 24 | 67.34 | 70.47 ± 0.95 | +18.34 ± 1.16 | 1,850 |
| **Solo Claude (portfolio pick)** | 32 | 74.60 | **75.74** ± 0.92 | +23.61 ± 1.21 | 1,992 |
| Solo Gemini | 0 | — | 52.13 (no edit) | 0 | 1,388 |

- **Specificity:** same-size random edits of the two picks scored 0.1 and 0.9. The policy is a hand-built logic
  circuit, so random changes destroy it, while the editors' edits are specific.
- **Selection:** both views (valid only, and as run) pick the same candidates.

| Comparison | Return difference | SE |
|---|---:|---:|
| **Peers − portfolio (primary)** | **−0.94** | 0.78 |
| Peers − OpenAI solo | +4.33 | 0.93 |
| Peers − Claude solo | −0.94 | 0.78 |
| Peers − Gemini solo | +22.67 | 1.19 |

## What this shows

1. **Weight editing works well on this task.** Every editor that tested an edit raised Astra's network from 52 to
   70–76, changing only 24–32 of 20,486 weights. That's about two-thirds of the way to PPO's 88.9. The edits
   strengthen the circuit's "danger two rows ahead" signal and add a gated escape-upward unit. Hits fell, and
   episodes lasted about 43% longer. Oxygen deaths rose from 67 to about 230, which leaves the next obvious fix.
2. **The council tied the best solo editor.** It didn't beat it: −0.9 ± 0.8 against Claude alone.
3. **Sharing again lifted the weaker member and made the members converge.** OpenAI scored 70.5 alone and 74.8 as a
   peer. The Gemini peer submitted the OpenAI peer's exact patch, so the three peers ended with essentially one
   edit. That's the same pattern as the routed digits and Spurious ImageNet.
4. **Gemini ran out of budget in both arms.** It used 1–2 of its 8 passes, spending its allowance on long analysis
   turns. This replicate therefore says little about Gemini as an editor.

## Audit

All six sessions are valid under the paper's rules. Three blocks were read closely:
- **Claude solo:** a truth table of one unit it designed, on hypothetical ±1 inputs.
- **Gemini peer:** a hand-built "player in row r" input propagated through the original first two layers, to read
  what each unit encodes.
- **OpenAI solo:** `tanh(3)` used as a constant to size a bias compensation.

None of them replays the network on recorded data, scores a candidate, or plays the game.
