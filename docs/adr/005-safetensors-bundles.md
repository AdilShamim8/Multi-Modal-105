# ADR-005: safetensors bundles as the deployment artifact

- **Status:** accepted (2026-10-18)
- **Context:** serving must never load raw training checkpoints: they
  carry optimizer state (3× size), no integrity contract, and no record
  of what produced them — "which FID did this file have?" becomes
  archaeology.
- **Decision:** the only deployable artifact is a **bundle**: weights in
  safetensors + `bundle.json` manifest (format, platform version, export
  timestamp, sha256, byte size). `agent/export.py` produces them;
  `read_manifest()` verifies checksums at load time.
- **Why safetensors (vs pickle/torchscript-only):** zero-copy mmap loads,
  no arbitrary-code-execution deserialization risk, and a stable format
  torchscript/onnx exports can sit alongside.
- **Consequences:**
  - (+) Corrupt or tampered artifacts fail at load, not mid-generation.
  - (+) Rollback = repoint the registry key; previous bundle is intact
    and verifiable.
  - (−) An extra export step between training and serving (by design:
    the eval gate runs at exactly that step).
