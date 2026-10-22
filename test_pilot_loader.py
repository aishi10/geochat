import torch
from transformers import AutoTokenizer, CLIPImageProcessor
from geochat.train.train import LazySupervisedDataset, DataArguments

MODEL = r"C:\Geochat\models\llava-v1.5-7b"
DATA = r"bigearthnet_pilot\pilot.json"
IMAGES = r"bigearthnet_pilot"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    model_max_length=2048,
    padding_side="right",
    use_fast=False,
)

print("Loading image processor...")
processor = CLIPImageProcessor.from_pretrained(
    "openai/clip-vit-large-patch14-336"
)

print("Creating GeoChat dataset...")
data_args = DataArguments(
    image_folder=IMAGES,
    is_multimodal=True,
    image_aspect_ratio="pad",
)
data_args.image_processor = processor
data_args.mm_use_im_start_end = False

dataset = LazySupervisedDataset(DATA, tokenizer, data_args)
print("Dataset size:", len(dataset))

print("Loading first example...")
sample = dataset[0]
print("Input IDs shape:", tuple(sample["input_ids"].shape))
print("Labels shape:", tuple(sample["labels"].shape))
print("Image tensor shape:", tuple(sample["image"].shape))
print("Image dtype:", sample["image"].dtype)
print("PIL/image preprocessing test passed.")

