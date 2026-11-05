"""Safetensors in numpy, with no torch: raw bit access, lazy reads and sharded checkpoints.

Derived from exp8_math/code/st_numpy.py (MIT) and the lazy reader of exp7_sycophancy/.../sandbox/prelude.py. Changes:
memory-mapped reads, so one tensor of a checkpoint larger than RAM can be read; sharded checkpoints; header validation;
casts that refuse non-finite results. bfloat16 tensors are handled as their raw uint16 bits.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64, "I64": np.int64, "I32": np.int32,
      "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}
FLOATS = ("BF16", "F16", "F32", "F64")
LONG_NAMES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64", "I32": "int32",
              "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool"}
SHORT_NAMES = {v: k for k, v in LONG_NAMES.items()}
MAX_HEADER = 1 << 28
_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def read_header(path: str | os.PathLike) -> tuple[dict, int]:
    """(header, offset of the data section). The header's size and tensor offsets are checked against the file."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if n > MAX_HEADER or 8 + n > size:
            raise ValueError(f"{path}: invalid header size {n}")
        head = json.loads(f.read(n))
    for name, info in head.items():
        if name == "__metadata__":
            continue
        a, b = info["data_offsets"]
        if not 0 <= a <= b <= size - 8 - n:
            raise ValueError(f"{path}: tensor {name} lies outside the file")
    return head, 8 + n


class SafetensorFile:
    """One .safetensors file. Tensors are read lazily through a read-only memory map."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        head, self.offset = read_header(self.path)
        self.metadata = head.pop("__metadata__", None)
        self.info = head
        self._raw = None

    def names(self) -> list[str]:
        return list(self.info)

    def raw(self, name: str) -> tuple[str, np.ndarray]:
        """(dtype string, read-only array). BF16 arrays are uint16 bit patterns."""
        if self._raw is None:
            self._raw = np.memmap(self.path, dtype=np.uint8, mode="r", offset=self.offset)
        i = self.info[name]
        if i["dtype"] not in DT:
            raise ValueError(f"{name}: unsupported tensor dtype {i['dtype']}")
        a, b = i["data_offsets"]
        return i["dtype"], self._raw[a:b].view(DT[i["dtype"]]).reshape(i["shape"])


class Checkpoint:
    """A model's weights: one .safetensors file, or a folder holding one or several shards."""

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self.files: dict[str, SafetensorFile] = {}
        self.where: dict[str, str] = {}
        if self.path.is_file():
            self.dir = self.path.parent
            shards = [self.path]
        elif self.path.is_dir():
            self.dir = self.path
            shards = self._shards()
        else:
            raise FileNotFoundError(self.path)
        for shard in shards:
            f = SafetensorFile(shard)
            self.files[shard.name] = f
            for name in f.names():
                self.where[name] = shard.name

    def _shards(self) -> list[Path]:
        indexes = sorted(self.dir.glob("*.safetensors.index.json"))
        if indexes:
            weight_map = json.loads(indexes[0].read_text())["weight_map"]
            return [self.dir / s for s in sorted(set(weight_map.values()))]
        shards = sorted(self.dir.glob("*.safetensors"))
        if not shards:
            raise FileNotFoundError(f"no .safetensors files in {self.dir}")
        return shards

    @property
    def config(self) -> dict:
        p = self.dir / "config.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def names(self) -> list[str]:
        return list(self.where)

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.files[self.where[name]].info[name]["shape"])

    def dtype(self, name: str) -> str:
        return self.files[self.where[name]].info[name]["dtype"]

    def raw(self, name: str) -> tuple[str, np.ndarray]:
        return self.files[self.where[name]].raw(name)

    def float(self, name: str) -> np.ndarray:
        """The tensor as a new float array: float32, or float64 for F64 tensors (bf16 -> float32 is exact)."""
        dtype, arr = self.raw(name)
        return to_float(dtype, arr)


def to_bits(arr: np.ndarray) -> np.ndarray:
    """The array's raw bits as an unsigned integer array of the same shape."""
    return np.ascontiguousarray(arr).view(_UINT[arr.dtype.itemsize])


def bf16_bits(x32: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit pattern, round to nearest even (as torch's .to(torch.bfloat16) for finite values)."""
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits).astype(np.uint32) << 16).view(np.float32)


def to_float(dtype: str, arr: np.ndarray) -> np.ndarray:
    if dtype == "BF16":
        return bf16_to_f32(arr)
    if dtype == "F64":
        return np.array(arr, dtype=np.float64)
    if dtype in FLOATS:
        return np.array(arr, dtype=np.float32)
    raise ValueError(f"not a floating-point tensor: {dtype}")


def cast_bits(x: np.ndarray, dtype: str) -> np.ndarray:
    """Round a float array to `dtype` and return its raw bits, without checks."""
    if dtype == "BF16":
        return bf16_bits(np.asarray(x, dtype=np.float32))
    return to_bits(np.asarray(x, dtype=DT[dtype]))


def cast_to(x: np.ndarray, dtype: str, name: str = "tensor") -> np.ndarray:
    """Raw bits of `x` rounded to `dtype`. Refuses values that are not finite before or after the cast: a value near the
    float32 maximum becomes infinity in bfloat16, and NaN does not survive bf16_bits."""
    if dtype not in FLOATS:
        raise ValueError(f"{name}: cannot cast to {dtype}")
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError(f"{name}: contains non-finite values")
    bits = cast_bits(x, dtype)
    if not np.isfinite(to_float(dtype, bits.view(DT[dtype]))).all():
        raise ValueError(f"{name}: values overflow {LONG_NAMES[dtype]}")
    return bits


def save(path: str | os.PathLike, tensors: dict[str, tuple[str, np.ndarray]], metadata: dict | None = None) -> None:
    """tensors: {name: (dtype string, array)}, written in name order."""
    head, off = {}, 0
    for name in sorted(tensors):
        dt, arr = tensors[name]
        n = arr.size * arr.dtype.itemsize
        head[name] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + n]}
        off += n
    if metadata:
        head["__metadata__"] = metadata
    h = json.dumps(head, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(h)))
        f.write(h)
        for name in sorted(tensors):
            f.write(np.ascontiguousarray(tensors[name][1]).tobytes())
