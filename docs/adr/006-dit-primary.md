# ADR-006: DiT as the primary denoiser; U-Net retained as reference

- **Status:** accepted (2026-10-18)
- **Context:** the repo implements both SD-style U-Net and DiT. Serving,
  scaling math, and the video extension all need ONE primary architecture
  to size against.
- **Decision:** DiT is the primary: the t2i production profile
  (`t2i-dit-s2`) and all T2V work target DiT. U-Net remains fully
  implemented, tested, and configurable (`configs/t2i/unet_base.yaml`)
  as the reference/comparison architecture.
- **Rationale:**
  - Transformer scaling laws are better characterized (depth/width
    knobs trade cleanly) — capacity planning is more honest on DiT.
  - The video path (factorized space-time attention, 3D RoPE) is a DiT
    native extension; a U-Net video variant would be a parallel universe
    of maintenance.
  - adaLN-Zero conditioning (timestep + text) removes the
    cross-attention bottleneck at high resolution.
- **Consequences:**
  - (+) One architecture to benchmark, scale, and harden.
  - (+) U-Net comparisons keep the "is DiT actually better here?"
    question answerable in-repo rather than by folklore.
  - (−) U-Net bug fixes are second-tier (documented, tests kept green,
    but no production dependency on it).
