import torch
from transformers import BitsAndBytesConfig
from geochat.model.language_model.geochat_llama import GeoChatLlamaForCausalLM

model_path = r"C:\Geochat\models\llava-v1.5-7b"

print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

print("GPU:", torch.cuda.get_device_name(0))
print("Loading model in 4-bit mode...")

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)

model = GeoChatLlamaForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.float16,
    quantization_config=quant_config,
    low_cpu_mem_usage=True,
)

print("Model loaded successfully.")
print("GPU memory allocated (GB):", round(torch.cuda.memory_allocated(0) / 1e9, 2))
