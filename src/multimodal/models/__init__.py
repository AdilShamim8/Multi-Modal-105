"""Generative model implementations.

Families: VAE (latent compression), GANs (adversarial synthesis),
autoregressive (VQ token models), diffusion (U-Net / DiT), plus the text
encoders and causal video compression networks used by the T2I/T2V
pipelines.

Subpackages (import directly — they pull in torch):
    multimodal.models.vae             AutoencoderKL, Encoder, Decoder
    multimodal.models.gans            StyleGAN2 generator/discriminator/system
    multimodal.models.autoregressive  VQ-VAE tokenizer
    multimodal.models.compression     causal 3D video autoencoder
    multimodal.models.diffusion       NoiseScheduler, U-Net, DiT, VideoDiT
    multimodal.models.text_encoder    CLIP/T5 conditioning encoders

This package is imported lazily by the platform core — never at API-process
startup unless a model is actually being loaded.
"""
