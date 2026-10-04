# Development environment

Recorded on 4 October 2026 before data download or training.

## Supported runtime

- Python: 3.12
- Environment and dependency manager: uv 0.12.17
- Dependency source of truth: `pyproject.toml` and `uv.lock`
- Default platform verified during setup: macOS on ARM64

## Development hardware

- Machine: MacBook Pro Mac16,8
- Processor: Apple M4 Pro, 12 CPU cores (8 performance and 4 efficiency)
- Memory: 48 GB unified memory
- Accelerator: integrated Apple M4 Pro GPU, 16 cores, Metal support
- Local storage at setup: 926 GiB total, approximately 540 GiB available

The installed PyTorch build includes the Metal Performance Shaders backend, but `torch.backends.mps.is_available()` returned `False` in the current execution environment. Treat CPU as the only verified compute target for now. The first compute pilot must recheck accelerator access and record whether CPU or MPS is used, peak memory, runtime, batch size, and any fallback operations.

## Local storage configuration

The default paths are `data/` and `artifacts/`. Set `VOICE_DETECTOR_DATA_DIR` and `VOICE_DETECTOR_ARTIFACTS_DIR` to place large files outside the checkout. Configuration files must remain credential-free.

## Compatibility check

Repository setup is accepted only after the locked environment installs, the package imports, the smoke command imports every major runtime dependency, and lint and unit tests pass. This does not validate data decoding, GPU training, or model accuracy.
