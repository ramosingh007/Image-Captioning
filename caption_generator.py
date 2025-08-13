from unsloth import FastVisionModel
from transformers import TextStreamer
from PIL import Image
import torch

# Load model and tokenizer
model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth"
)
FastVisionModel.for_inference(model)  # Set model to inference mode

# ---------- CONFIG ----------
image_path = "class.jpeg"  # Local image path
instruction = "Describe this image in detail for a blind user."
output_file = "captions.txt"
# ----------------------------

# Load and resize image to reduce memory usage
image = Image.open(image_path).convert("RGB")
image = image.resize((512, 512))  # Resize to 512x512

# Prepare message for tokenizer
messages = [
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": instruction}
    ]}
]

# Apply chat template to build prompt
input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

# Tokenize input
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

# Generate caption (reduced token count for memory efficiency)
generated_tokens = model.generate(
    **inputs,
    streamer=None,  # Set to TextStreamer(tokenizer) to see live output
    max_new_tokens=100,  
    use_cache=True,
    temperature=0.8,
    top_p=0.9
)

# Decode tokens to get final caption
caption = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()

# Save to file
with open(output_file, "a", encoding="utf-8") as f:
    f.write(f"Image File: {image_path}\n")
    f.write(f"Caption: {caption}\n")
    f.write("-" * 80 + "\n")

print("✅ Caption generated and saved to file!")
