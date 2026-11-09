"""The background process behind `wsplab session`: a normal run whose editor is whoever talks to its socket."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

CLI_NOTE = ("You are working through the wsplab command line, so read this text accordingly. A ```python block is `wsplab session "
            "exec`, a ```forward block is `wsplab session forward`, your explanation is their --why, and replying without a "
            "block is `wsplab session finish`. Unlike the session described above, you may finish with forward passes unused: "
            "do so once your edit is frozen and confirmed, rather than spending passes after seeing HOLDOUT.")


def main(folder: str) -> None:
    import wsplab

    from ..cli import load_scorer, parse_build
    from ..editors.agent import AgentEditor
    from ..kernel import Limits
    from .driver import Policy

    root = Path(folder)
    c = json.loads((root / "agent.json").read_text())
    editor = AgentEditor(root)
    policy = Policy(handoff_every=10**9, require_all_passes=False)
    note = "\n\n".join(x for x in (c.get("note"), CLI_NOTE) if x)
    limits = Limits(seconds=c["run_seconds"], memory_gb=c.get("memory_gb"))
    try:
        result = wsplab.run(c["model"], build=parse_build(c.get("build")), description=c["description"], note=note,
                            mode=c["mode"], practice=load_scorer(c.get("practice"), c["metric"]),
                            evaluate=load_scorer(c.get("eval"), c["metric"], "eval"), editor=editor, budget=c["budget"],
                            folder=root / "run", policy=policy, limits=limits, no_eval=c.get("no_eval", False),
                            sandbox=c.get("sandbox", "bwrap"), semantic=c.get("semantic", False))
        editor.finished({"verdict": result.verdict, "reasons": result.reasons, "baseline": result.baseline, "final": result.final,
                         "patch": str(result.patch) if result.patch else None, "report": str(result.folder / "report.md")})
    except Exception as e:  # noqa: BLE001 - whatever goes wrong must reach the agent instead of hanging it
        editor.finished({"verdict": "error", "reasons": [f"{type(e).__name__}: {e}", traceback.format_exc()[-1500:]]})
    import time

    time.sleep(30)


if __name__ == "__main__":
    main(sys.argv[1])
