# Model Registry Guide

> Last audited: 2026-10-18 · Code: `agent/export.py`, `scripts/preload_models.py`

## What a registry entry is

```
$MM_MODEL_DIR/
└── t2i-dit-s2/                 # registry key = directory name
    ├── bundle.json             # manifest: format, versions, sha256, bytes
    ├── weights.safetensors     # training weights (fp16)
    └── model.torchscript       # serving artifact
```

The manifest is the contract:

```json
{
  "format": "torchscript",
  "platform_version": "1.0.0",
  "release_date": "2026-10-18",
  "exported_at": "2026-10-18T15:04:11Z",
  "artifact": "model.torchscript",
  "sha256": "9f2c…",
  "bytes": 273618124
}
```

## Lifecycle

```
train → eval gates (FID/IS/CLIP) → mma export → registry volume
      → workers load on next restart (checksum-verified)
```

1. **Export** — `mma export --model-dir <run>/final --format torchscript`.
   Only EMA weights leave training (ADR-005).
2. **Promote** — copy/sync the bundle directory into the registry under its
   key. The preload Job verifies checksums before workers ever see it.
3. **Activate** — change `MM_T2I_MODEL` (ConfigMap) + roll workers. The
   previous bundle stays — rollback is a repoint, not a re-export.
4. **Retire** — delete after two release cycles of non-use; manifests keep
   the audit trail in git regardless.

## Rules

* **No raw checkpoints in the registry.** Optimizer state triples the size
  and carries zero serving value.
* **Registry keys are immutable.** A new bundle for the same capability
  gets a new version suffix; overwriting an existing key breaks rollback.
* **Checksums are load-time gates, not paperwork** — `read_manifest()`
  refuses corrupted or tampered artifacts (tested in
  `tests/unit/test_export.py`).
* **Bundles carry their config** — `config.json` beside weights records
  architecture params so the loader never guesses.
