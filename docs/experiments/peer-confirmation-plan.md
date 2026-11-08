# Fresh-seed two-provider confirmation

The seed-21 pilot produced a numerical advantage (91.76% peer winner versus 75.88%
completed terra solo), but it does not qualify as clean proof. Gemini replayed a
modified network tail in Python in tool record 6, and several sessions hit cost
or context guards. The initial byte-count guard also stopped runs prematurely.
Do not erase these runs or reinterpret them as successful confirmation.

The user requested continued work to find a hard case. This second phase stays
inside the previously authorized per-provider budgets, rather than requesting new
credit. Its maximum estimated allocation is $8.50: $0.75 OpenAI and $3.50 Anthropic
per arm. Before launch, compare these reservations with original ledgers plus all
current experiment ledgers. Gemini gets no new model calls: long responses with
zero reported output made the initial $12.50 pilot ceiling impossible to certify.
The first stage reports $5.97 but has up to $6.55 of unreported output exposure if
all requested token limits were consumed; the tiny usage check is additional.

Fixed before calls:

- Fresh construction seed 22; unchanged calibrated task and scoring.
- Two arms: terra + Claude peers; the same two independent solo models.
- Both use high effort, 40 Python runs, 6 forward passes, 60 turns, identical
  early-finish policy and candidate-selection data.
- Identical clarification in both prompts: do not recompute modified network
  predictions in Python, including partial tails from saved activations. Use the
  official forward tool. Save a plausible candidate early.
- Same corrected guard in both arms. Use reported tokens for stable prefixes,
  bound new text conservatively, reduce output allowance near the monetary cap.
  Nonempty output with zero reported tokens is charged at the requested maximum
  and flagged as estimated, not silently counted as free.
- Compare selection-chosen candidates on identical held-out queries. Report all
  stopping conditions and individual solo diagnostics. Audit every contributing
  peer for counterfactual simulation or fitting, not just the selected model.
- A numerical gain does not pass if protocol violations or prematurely stopped
  solos explain it. Even a clean win is one fresh task seed, not general proof.

No changes based on this phase's scores; retain the outcome either way.
