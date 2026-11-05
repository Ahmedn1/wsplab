"""Generate the base prompt from the authors' last prompt revision (v5) by replacing exact substrings with slots.

Every replacement states how many times it must match, so a wrong slot fails loudly. tests/conformance checks the
reverse: rendering the template with the Gemma values gives the original text byte for byte.
"""

import json
import re
from pathlib import Path

from wsplab import upstream

ROOT = Path(__file__).resolve().parent.parent
v5 = upstream.fetch("exp8_math/gemma_followup/runs/g50b/turns/t001/prompt.txt").read_text()

network = v5[len("TASK. You are given the weights of "):v5.index(" Make it better at mathematics")].removesuffix(".")
task = v5[v5.index("Make it better at mathematics"):v5.index("\n\nEVALUATION.")]
protocol = v5[v5.index("fixed protocol: ") + len("fixed protocol: "):v5.index(". Only weight values")]
practice = v5[v5.index("2. Practice problems") + 3:v5.index("\n3. A forward-pass tool")]

SLOTS = [
    (network, "{{network}}", 1, network),
    (task, "{{task_sentence}}", 1, task),
    (protocol, "{{protocol}}", 1, protocol),
    (practice, "{{practice}}", 1, practice),
    ("evaluated on MATH-500, which", "evaluated on {{benchmark}}, which", 1, "MATH-500"),
    ("Out of scope: the final norm and the embedding / output matrix (save_edit",
     "Out of scope: {{out_of_scope}} (save_edit", 1, "the final norm and the embedding / output matrix"),
    ("numpy, scipy, scikit-learn, PyTorch, pandas, safetensors, tokenizers", "{{libraries}}", 1,
     "numpy, scipy, scikit-learn, PyTorch, pandas, safetensors, tokenizers"),
    ("Up to 900 s and 96 GB per run, output cut at 10,000 characters, at most 300 runs.",
     "Up to {{run_seconds}} s and {{run_memory}} per run, output cut at {{output_cut}} characters, at most {{max_runs}} runs.",
     1, None),
    ("with a budget of 50 passes", "with a budget of {{passes}} passes", 1, None),
    ("until all 50 forward passes", "until all {{passes}} forward passes", 1, None),
    ('"max_tokens": at most 8,192', '"max_tokens": at most {{max_tokens}}', 1, None),
    ('"layers": [0..18]', '"layers": [0..{{n_layers}}]', 1, None),
    ('"hidden_states": [0..18]', '"hidden_states": [0..{{n_layers}}]', 1, None),
    ('at most 2,048 answers ("both": 1,024 per model)', 'at most {{gen_max}} answers ("both": {{both_max}} per model)', 1, None),
    ("the answer's first 8,192 tokens", "the answer's first {{act_tokens}} tokens", 1, None),
    ("up to 1,024 texts of up to 8,192 tokens (524,288 in total)",
     "up to {{tf_texts}} texts of up to {{tf_tokens}} tokens ({{tf_total}} in total)", 1, None),
    ("whether it reads 2 texts or 1,024:", "whether it reads 2 texts or {{tf_texts}}:", 1, None),
    ("a full generation pass takes several minutes", "a full generation pass takes {{pass_time}}", 1, None),
]

text = v5
values = {}
for old, new, count, value in SLOTS:
    assert text.count(old) == count, (old[:60], text.count(old))
    text = text.replace(old, new)
    if value is not None:
        values[re.search(r"\{\{(\w+)\}\}", new).group(1)] = value
values.update(run_seconds="900", run_memory="96 GB", output_cut="10,000", max_runs="300", passes="50",
              max_tokens="8,192", n_layers="18", gen_max="2,048", both_max="1,024", act_tokens="8,192",
              tf_texts="1,024", tf_tokens="8,192", tf_total="524,288", pass_time="several minutes")
(ROOT / "src/wsplab/prompts/templates/lm_edit.txt").write_text(text)
(ROOT / "tests/conformance/gemma_v5_slots.json").write_text(json.dumps(values, indent=1, ensure_ascii=False) + "\n")
print(len(v5), "->", len(text), "chars;", len(values), "slots")
