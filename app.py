import gradio as gr
from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch

# Load model and processor
model_name = "prithivMLmods/facial-age-detection"  # Update with actual model name on Hugging Face
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated label mapping
id2label = {
    "0": "age 01-10",
    "1": "age 11-20",
    "2": "age 21-30",
    "3": "age 31-40",
    "4": "age 41-55",
    "5": "age 56-65",
    "6": "age 66-80",
    "7": "age 80 +"
}

def classify_image(image):
    image = Image.fromarray(image).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction

# Gradio Interface
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=8, label="Age Group Classification"),
    title="Facial Age Detection",
    description="Upload a face image to estimate the age group: 01–10, 11–20, 21–30, 31–40, 41–55, 56–65, 66–80, or 80+."
)

if __name__ == "__main__":
    iface.launch()
