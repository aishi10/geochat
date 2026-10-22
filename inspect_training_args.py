import torch

p = r"C:\Geochat\geochat\output\bigearthnet_1000_3epochs_lora\checkpoint-3000\training_args.bin"
args = torch.load(p, map_location="cpu", weights_only=False)

for key in [
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "gradient_accumulation_steps",
    "model_max_length",
    "freeze_backbone",
    "tune_mm_mlp_adapter",
    "vision_tower",
    "image_aspect_ratio",
    "lora_enable",
    "lora_r",
    "lora_alpha",
]:
    print(f"{key}: {getattr(args, key, 'NOT STORED')}")
