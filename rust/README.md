# Rust port of `trm.doppler`'s numerical core

This directory contains a Rust workspace ported from the numerically
intensive parts of the original `trm/doppler/doppler.cc` CPython extension.

## Crates

- **`doppler-core`**: pure-Rust, dependency-light numerical core.
  - `op()` / `tr()`: image<->data projection routines (ports of the C++
    `op`/`tr` functions), including FFT-based instrumental-resolution
    blurring, now implemented with [`rustfft`](https://docs.rs/rustfft) /
    [`realfft`](https://docs.rs/realfft) instead of `fftw3`, and
    [`rayon`](https://docs.rs/rayon) data-parallel loops instead of OpenMP
    `#pragma` directives.
  - `gaussdef()` / `compute_default()`: separable 3-axis Gaussian image
    blurring and the per-image "default"/prior-image computation (ports of
    the C++ `gaussdef()` function and the shared body of
    `doppler_comdef()`/`doppler_memit()`'s mode-30 step), used both
    standalone (`comdef`) and inside MEM iteration (recomputing the
    default each iteration when any image has a non-uniform default or a
    bias != 1).
  - See `doppler-core/tests/adjoint.rs` for a dot-product ("adjoint") test
    verifying `tr` is the mathematical transpose of `op`, the standard
    correctness check for tomographic projection/back-projection operator
    pairs used in MEM reconstruction, and `doppler-core/tests/gaussdef.rs`
    for flux-conservation/sanity checks on the blurring routines.

- **`doppler-mem`**: MEM iteration driver (`run_memit()`), bridging
  `doppler-core`'s `op`/`tr` to the external `memsys` crate's
  `MemTransform` trait (see below), mirroring `doppler_memit()`'s mode
  selection and convergence test. `DopplerImages`/`DopplerData` hold the
  owned, flattened Doppler map/dataset description needed across repeated
  MEM iterations; `DopplerTransform` implements `memsys::MemTransform` in
  terms of `doppler_core::op`/`tr`.

- **`memsys-stub`** (package name `memsys`): a **temporary local
  placeholder** for the real
  [`trm-astro/rust-mem`](https://github.com/trm-astro/rust-mem) crate,
  pending it publishing a proper library API. See its own doc comment
  (`memsys-stub/src/lib.rs`) for the full rationale; in short:
  - `rust-mem`'s `main` branch is not currently usable as a normal Cargo
    dependency (no `src/lib.rs`; `opus`/`tropus` are resolved via a
    hardcoded `crate::opus::opus`/`crate::tropus::tropus` module path that
    only works when its `memsys.rs` is vendored directly into the
    *consuming* crate).
  - We agreed on a design with its maintainer to fix this: add
    `src/lib.rs`, decouple the map-space/data-space array sizes (currently
    both forced to a single `ndata`), and replace the hardcoded
    `opus`/`tropus` paths with a `MemTransform` trait injected into
    `memprm()`.
  - `memsys-stub` implements that *exact* agreed public API
    (`MemGblst`/`MemTransform`/`memprm`) so `doppler-mem` can be written
    and integration-tested today. **It is not a port of the real MEM
    (Skilling & Bryan) algorithm** -- `memprm()` here is a simple,
    unguarded gradient step, sufficient to validate the plumbing
    end-to-end (see `doppler-mem/tests/memit.rs`) but not to produce
    scientifically meaningful reconstructions or diagnostics.
  - Once `trm-astro/rust-mem` publishes the real API, swap the `memsys`
    path dependency in `doppler-mem/Cargo.toml` for a git dependency on
    that crate; because the public API is designed to match exactly, no
    other code should need to change.

- **`doppler-py`**: thin [PyO3](https://pyo3.rs) bindings exposing `op`,
  `tr`, `default_image` and `memit` to Python as a compiled extension
  module, taking numpy arrays and plain Python lists instead of reaching
  into `Map`/`Data` objects via the Python C API (unlike the original
  `doppler.cc`, which did its own attribute-walking of `Map`/`Data`
  objects). It is built as `trm.doppler._doppler_rs` -- nested inside the
  main Python package, not a standalone top-level module -- by the
  `[tool.maturin]` section of the repository root's `pyproject.toml`; the
  pure-Python `trm/doppler/_rust.py` imports it with `from . import
  _doppler_rs` and marshals `Map`/`Data` objects into calls to it.

## Building

For normal use, build/install everything (the Python package and the
Rust extension together) from the repository root:

```
pip install -e .          # or: pip install .
```

This uses `maturin` as the PEP 517 build backend (see the root
`pyproject.toml`) to compile `rust/doppler-py` and place it at
`trm.doppler._doppler_rs`; no separate build step is needed, and (unlike
the retired C++ `doppler.cc` extension) no external `fftw`/`cpp-mem`
C/C++ libraries are required.

For working on the Rust code in isolation:

```
cd rust
cargo build -p doppler-core -p doppler-mem   # pure Rust libraries + tests
cargo test  -p doppler-core -p doppler-mem

cd doppler-py
maturin build --release       # builds the trm.doppler._doppler_rs wheel directly
```

Note: `cargo build -p doppler-py` (or `--workspace`) will fail to *link*
because `pyo3`'s `extension-module` feature intentionally does not link
against `libpython` (symbols are resolved by the embedding interpreter at
import time). Use `maturin build`/`maturin develop`, or `pip install -e .`
from the repository root, to build/install the Python extension; use
`cargo check -p doppler-py` to type-check without linking.

## Status / follow-ups

This covers the numerical core (`op`/`tr`), default-image computation
(`gaussdef`/`compute_default`), and a MEM iteration driver (`run_memit`),
thin PyO3 wrappers matching the calling convention of the corresponding
C++ functions (`op`/`tr`, `comdef`'s per-image default computation, and
`memit`), and a Python glue module,
[`trm/doppler/_rust.py`](../trm/doppler/_rust.py), providing `comdat`,
`comdef`, `datcom` and `memit` functions matching the original C++
extension's names/calling convention but operating directly on this
package's pure-Python `Map`/`Data` objects (flattening/unflattening them
into the flat arrays `_doppler_rs` expects, in plain Python rather than
the Python C API). Its flattening has been verified end-to-end, including
a dot-product ("adjoint") check through the full `Map`/`Data` -> Rust ->
`Map`/`Data` round trip, confirming the array-shape/axis conventions match
the original C++ exactly.

The legacy C++ extension (`trm/doppler/doppler.cc`) and its `setup.py`
build have been retired and removed; `trm/doppler/__init__.py` now
imports `comdat`/`comdef`/`datcom`/`memit` from `_rust.py` instead, so
existing scripts (`trm/doppler/scripts/*.py`) that call
`doppler.comdat(...)` etc. did not need any changes.

Not yet ported:


- Real MEM iteration: pending `trm-astro/rust-mem` publishing its agreed
  library API (see the `memsys-stub` note above). `doppler-mem`'s
  `run_memit()` and PyO3's `memit()` binding are structured so that
  swapping in the real `memsys` crate should require no further changes
  to this repository.
