def get_model_config():
    return {
        "model_type": "diffusion_cond",
        "sample_size": 2097152,
        "sample_rate": 44100,
        "audio_channels": 2,
        "model": {
            "pretransform": {
                "type": "autoencoder",
                "iterate_batch": True,
                "config": {
                    "encoder": {
                        "type": "oobleck",
                        "requires_grad": False,
                        "config": {
                            "in_channels": 2,
                            "channels": 128,
                            "c_mults": [1, 2, 4, 8, 16],
                            "strides": [2, 4, 4, 8, 8],
                            "latent_dim": 128,
                            "use_snake": True
                        }
                    },
                    "decoder": {
                        "type": "oobleck",
                        "config": {
                            "out_channels": 2,
                            "channels": 128,
                            "c_mults": [1, 2, 4, 8, 16],
                            "strides": [2, 4, 4, 8, 8],
                            "latent_dim": 64,
                            "use_snake": True,
                            "final_tanh": False
                        }
                    },
                    "bottleneck": {
                        "type": "vae"
                    },
                    "latent_dim": 64,
                    "downsampling_ratio": 2048,
                    "io_channels": 2
                }
            },
            "conditioning": {
                "configs": [
                    {
                        "id": "prompt",
                        "type": "t5",
                        "config": {
                            "t5_model_name": "t5-base",
                            "max_length": 128
                        }
                    },
                    {
                        "id": "seconds_start",
                        "type": "number",
                        "config": {
                            "min_val": 0,
                            "max_val": 512
                        }
                    },
                    {
                        "id": "seconds_total",
                        "type": "number",
                        "config": {
                            "min_val": 0,
                            "max_val": 512
                        }
                    }
                ],
                "cond_dim": 768
            },
            "diffusion": {
                "cross_attention_cond_ids": ["prompt", "seconds_start", "seconds_total"],
                "global_cond_ids": ["seconds_start", "seconds_total"],
                "type": "dit",
                "config": {
                    "io_channels": 64,
                    "embed_dim": 1536,
                    "depth": 24,
                    "num_heads": 24,
                    "cond_token_dim": 768,
                    "global_cond_dim": 1536,
                    "project_cond_tokens": False,
                    "transformer_type": "continuous_transformer"
                }
            },
            "io_channels": 64
        },
        "training": {
            "use_ema": True,
            "log_loss_info": False,
            "optimizer_configs": {
                "diffusion": {
                    "optimizer": {
                        "type": "AdamW",
                        "config": {
                            "lr": 5e-5,
                            "betas": [0.9, 0.999],
                            "weight_decay": 1e-3
                        }
                    },
                    "scheduler": {
                        "type": "InverseLR",
                        "config": {
                            "inv_gamma": 1000000,
                            "power": 0.5,
                            "warmup": 0.99
                        }
                    }
                }
            },
            "demo": {
                "demo_every": 2000,
                "demo_steps": 250,
                "num_demos": 4,
                "demo_cond": [
                    {"prompt": "Amen break 174 BPM", "seconds_start": 0, "seconds_total": 12},
                    {"prompt": "A beautiful orchestral symphony, classical music", "seconds_start": 0, "seconds_total": 160},
                    {"prompt": "Chill hip-hop beat, chillhop", "seconds_start": 0, "seconds_total": 190},
                    {"prompt": "A pop song about love and loss", "seconds_start": 0, "seconds_total": 180}
                ],
                "demo_cfg_scales": [3, 6, 9]
            }
        }
    }
