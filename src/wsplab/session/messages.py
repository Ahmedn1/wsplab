"""What the harness says to the editor between turns.

The wording comes from the logged runs, which differ slightly by prompt version, so a Messages object is a replaceable
pack. V3 is the wording of exp8_math/runs/p50c, V5 of exp8_math/gemma_followup/runs/g50b. The nudge and block-count
messages never occurred in the released logs and are new.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

EXPLAIN_V3 = ("EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about "
              "3 to 8 sentences): what the last result showed, what you conclude from it, what you will do now and why, "
              "and what outcome would confirm or refute your current hypothesis. In your final answer, also summarize "
              "the path that led to your edits. These explanations are read by the experimenters.")
EXPLAIN_V5 = EXPLAIN_V3


@dataclass(frozen=True)
class Messages:
    python_output: str = "OUTPUT of Python run {k} of at most {limit}:\n```\n{out}\n```"
    pass_done: str = ("Forward pass {k} of {budget} done ({kind}); results in {where}. Summary:\n```\n{summary}\n```\n"
                      "Forward passes left: {left}.")
    pass_edits: str = "Edits used in this pass: {names} (edit hash {digest})."
    all_passes_used: str = ("All forward passes are used. Make your final choice of edits (Python is still available), "
                            "then reply without a code block.")
    pass_failed: str = "The forward pass failed and was not counted: {error}"
    pass_invalid: str = "The forward request is not valid ({error}). Not counted."
    no_explanation: str = ("Your last reply had no explanation, so its block was NOT run (nothing was used). {explain} "
                           "Reply again: first your explanation, then the block you want to run now (it may be the same "
                           "one).")
    explain: str = EXPLAIN_V5
    reminder: str = "(Reminder: before your next block, explain what this result shows, what you will do next and why.)"
    refused: str = ("[The output of your last step could not be delivered: the provider's content filter refused the "
                    "message. The model's vocabulary contains offensive and adult strings, and listing raw vocabulary "
                    "tokens can trigger this. Repeat what you need, but print only tokens relevant to your question "
                    "(digits, math symbols, common math words) or token ids and numbers, not raw lists of top tokens. The "
                    "filter can also refuse messages that contain long reasoning transcripts (many generated or "
                    "hand-written 'Wait, ...' passages): print short excerpts (at most about 300 characters each) and "
                    "statistics instead of whole texts.]")
    note: str = "[note from the experimenters] {text}"
    new_rule: str = "[note from the experimenters: a new instruction for the rest of the session] {text}"
    kernel_restarted: str = "[note: the harness was restarted, so your kernel restarted too and its variables are gone]"
    handoff: str = ("THIS IS A RESTARTED CONVERSATION (to keep it short): {replies} replies happened before. Forward "
                    "passes used: {passes} of {budget}; Python runs used: {runs} of {limit}. Your kernel is still running "
                    "with its variables. Tensors currently edited: {edited}. Files in /work: {files}.\n"
                    "YOUR NOTEBOOK (/work/notes.md, last {notes_chars:,} characters):\n```\n{notes}\n```\n"
                    "Continue from here.\n\nLast result:\n{last}")
    # New: these situations never occurred in the released logs.
    nudge: str = "Forward passes left: {left}. The session continues until all of them are used: reply with a block."
    runs_exhausted: str = "No Python runs are left (limit {limit}), so this block was NOT run."
    too_many_blocks: str = "Your reply had {n} blocks, so none was run. Reply with exactly one block."
    passes_exhausted: str = "No forward passes are left, so this block was NOT run."
    truncated: str = "Your reply hit the output limit and was cut off, so nothing in it was run. Reply again, more briefly."


V5 = Messages()
V3 = replace(
    V5,
    no_explanation=("Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). "
                    "{explain} Send the block again, preceded by your explanation."),
    explain=EXPLAIN_V3,
)
