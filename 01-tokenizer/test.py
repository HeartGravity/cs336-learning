import wandb

wandb.init(
    project="cs336-a5-sft-v2",
    entity="heartgravity-org",
    name="wandb_sft",
    config={
        "model": "Qwen2.5-Math-1.5B",
        "dataset_tag": "raw", #raw, sf,grpo
        "batch_size": 64,
        "max_examples": 1000,
        "seed": 2026,
        "learning_rate": 2e-5,
    }
)