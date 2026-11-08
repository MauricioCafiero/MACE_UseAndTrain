"""Write a CPU copy (<model>.cpu.model) of a MACE model saved on a CUDA box.

mace_convert_device can't do this on a CUDA-less machine: e3nn's compiled
sub-modules call torch.jit.load(buffer) inside __setstate__ with no
map_location, so they try to rebuild on CUDA. Pin that call to CPU while
loading. The original file is left untouched.

    KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python longrange/to_cpu.py <model.model>
"""
import sys
from pathlib import Path

import torch

_jit_load = torch.jit.load
torch.jit.load = lambda f, map_location=None, **kw: _jit_load(f, map_location="cpu", **kw)

src = Path(sys.argv[1])
model = torch.load(src, map_location="cpu", weights_only=False)
torch.jit.load = _jit_load
dst = src.with_suffix(".cpu.model")
torch.save(model.to("cpu"), dst)
print("wrote", dst, "r_max =", float(model.r_max))
