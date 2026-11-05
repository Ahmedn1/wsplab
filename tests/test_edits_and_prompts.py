"""The `edits` language, and the promise that prompts differ only where they must."""

import numpy as np
import pytest

from wsplab.io import Checkpoint, save
from wsplab.passes.edits import apply_edits, last_block, render_weights, statements
from wsplab.prompts import PromptBuilder
from wsplab.prompts.builder import load_template, split_sections


def loader(**tensors):
    return lambda name: tensors[name]


def test_prompt_examples_of_the_edit_language():
    block = """W1[12, 0] = 2.5                     # one entry
W1[0:6, 3] = [1, 0, -1, 0, 2, 0]
b1[0:4] = -1
W1[3:600:60, 0] = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
W2[0:2, 0:3] = [[1, -1, 0.5], [0, 2, 0]]
W3[:, 2] *= 2
"""
    r = apply_edits(block, loader(W1=np.zeros((600, 32)), b1=np.zeros(32), W2=np.zeros((32, 32)), W3=np.ones((32, 6))))
    assert r.errors == [] and r.statements == 6
    assert r.tensors["W1"][12, 0] == 2.5 and r.tensors["W1"][4, 3] == 2 and r.tensors["b1"][3] == -1
    assert r.tensors["W1"][543, 0] == 1 and r.tensors["W2"][1, 1] == 2 and r.tensors["W3"][0, 2] == 2


def test_dotted_names_and_add_subtract():
    r = apply_edits("actor0.W[1, 1] += 3\nactor0.W[1, 1] -= 1", loader(**{"actor0.W": np.zeros((2, 2))}))
    assert r.tensors["actor0.W"][1, 1] == 2


def test_lists_may_continue_over_lines_and_keep_space_after_brackets():
    """The Seaquest and Asterix replies wrote `[ [ ... ], ... ]`, which the authors' first reader rejected."""
    block = "W[0:2, 0:2] = [ [1, 2],\n  [3, 4] ]"
    assert statements(block) == ["W[0:2, 0:2] = [ [1, 2], [3, 4] ]"]
    assert apply_edits(block, loader(W=np.zeros((2, 2)))).tensors["W"][1, 0] == 3


def test_bad_statements_are_skipped_and_the_rest_apply():
    r = apply_edits("W[0, 0] = 1\nnonsense\nW[9, 9] = 2\nV[0] = 1\nW[1, 1] = 3", loader(W=np.zeros((2, 2))))
    assert r.statements == 2 and len(r.errors) == 3 and r.tensors["W"][1, 1] == 3


@pytest.mark.parametrize("value", ["1e7", "__import__('os').system('true')", "float('inf')"])
def test_values_are_limited_and_never_executed(value):
    r = apply_edits(f"W[0, 0] = {value}", loader(W=np.zeros((2, 2))))
    assert r.statements == 0 and len(r.errors) == 1


def test_only_the_last_block_is_read():
    assert last_block("```edits\nA[0] = 1\n```\nchanged my mind\n```edits\nB[0] = 2\n```").strip() == "B[0] = 2"
    assert last_block("no block") is None


def test_render_weights(tmp_path):
    save(tmp_path / "m.safetensors", {"w": ("F32", np.array([[1.23456, 2], [3, 4]], np.float32))})
    text = render_weights(Checkpoint(tmp_path / "m.safetensors"))
    assert text.startswith("w, shape 2 x 2:\n1.235 2.000\n3.000 4.000")


def sections(name, **kw):
    return {s.split(".")[0]: s for s in split_sections(load_template(name, **kw))}


@pytest.mark.parametrize("name", ["generic_edit", "generic_write", "lm_write"])
def test_shared_sections_are_the_papers_text_everywhere(name):
    base, other = sections("lm_edit"), sections(name)
    assert other["EXPLAIN EVERY STEP"] == base["EXPLAIN EVERY STEP"]
    assert other["RULES"] == base["RULES"]


def test_kinds_differ_only_in_tools_and_method():
    a, b = sections("lm_edit"), sections("generic_edit")
    assert [k for k in a if a[k] != b.get(k)] == ["TOOLS", "METHOD"]


@pytest.mark.parametrize("name", ["lm_read", "generic_read", "inline_read"])
def test_read_prompts_never_mention_editing(name):
    text = load_template(name)
    assert "save_edit" not in text and "your edits" not in text and "any direct edit" not in text


def test_no_passes_removes_the_tool_and_says_so():
    text = load_template("generic_edit", passes=False)
    assert "forward-pass tool" not in text and "There are no forward passes" in text
    assert "{{passes}}" not in text and "{{modules}}" not in text


def test_generic_prompt_builds_with_no_frozen_tensors():
    slots = dict(network="n", benchmark="b", protocol="p", libraries="l", run_seconds=1, run_memory="1 GB", output_cut="1",
                 max_runs=1, practice="d", passes=1, modules="m", out_of_scope=None)
    text = PromptBuilder("generic_edit", slots=slots).build()
    assert "save_edit refuses them" not in text and "Out of scope: multiplying a whole tensor" in text
