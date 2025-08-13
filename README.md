# 🖼️ Image Captioning with Vision-Language LLM

This project uses **Unsloth's Qwen2.5-VL-7B-Instruct** model to generate captions from images — designed especially for accessibility use cases like describing images for blind users.

---

## 🚀 Features

- Caption any image using a local vision-language model
- Uses the [Unsloth](https://unsloth.ai) library for fast inference
- Outputs stored in `captions.txt`

---

## 📂 Folder Structure

image-captioning/
│
├── caption_generator.py # Main script to run
├── requirements.txt # Needed packages
├── .gitignore # Ignore unnecessary files
├── README.md # Project overview
└── captions.txt # Output log (auto-created)


---

## 🧠 How It Works

1. Loads a vision-language model (`Qwen2.5-VL`)
2. Accepts a local image (`class.jpeg`)
3. Prompts the model to describe the image
4. Saves captioned results to `captions.txt`

---

## 🛠️ Run It Locally

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Run the script
python caption_generator.py
