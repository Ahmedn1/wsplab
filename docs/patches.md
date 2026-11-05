# Patches

A patch is a folder.

| File | Contents |
|---|---|
| `edit.npz` | The changes |
| `manifest.json` | Per tensor: shape, dtype, form, hashes of the base and edited tensor |
| `patch.json` | The base model, evidence, verdict, editor, prompt and transcript hashes, and digests of the other two files |

`edit.npz` and `manifest.json` are the format of the authors' releases. wsplab reads theirs unchanged, and has checked its output
against every published hash. wsplab's own version adds other dtypes, a `dense` form and 64-bit indices.

Tensors are stored in the smallest exact form:

- `sparse`: changed entries as indices and raw bits.
- `lowrank`: factors plus the few entries where rounding differs.
- `dense`: the whole tensor.

## Commands

```sh
wsplab patch create --base model/ --submission work/edited --out patch
wsplab patch verify patch --base model/
wsplab patch apply patch --base model/ --out edited/
wsplab patch inspect patch
wsplab patch stack a b --base model/ --out both
wsplab patch import released/edit --out patch --name my-patch
wsplab patch push patch --repo user/name
wsplab patch pull user/name --out patch
```

`apply` needs only numpy. It overwrites the changed tensors' bytes in a copy of their shard and hard-links everything else, so
do not modify the output's untouched files in place, or use `--copy`.

`stack` combines patches. Patches on different tensors always combine. Two patches on one tensor must be chained, the later one
made on the result of the earlier one, and the hashes prove it.

## Trust

A patch is untrusted input. It is never executed, arrays load with pickling off, names and indices are checked, archives are
bounded before decompression, and any change to the files is caught by their digests.

None of that says a patch is good. A patch could hide a backdoor, and its evidence is only a claim by whoever made it. Measure it
yourself with `wsplab eval --patch` on data you trust. Respect the base model's license too: a patch is a derivative of it.
