
![467.png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/LA6do4hgVi-zNarLEhVGR.png)

# facial-age-detection

> facial-age-detection is a vision-language encoder model fine-tuned from `google/siglip2-base-patch16-512` for **multi-class image classification**. It is trained to detect and classify human faces into **age groups** ranging from early childhood to elderly adults. The model uses the `SiglipForImageClassification` architecture.

> \[!note]
> SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features
> [https://arxiv.org/pdf/2502.14786](https://arxiv.org/pdf/2502.14786)

```py
Classification Report:
              precision    recall  f1-score   support

   age 01-10     0.9614    0.9669    0.9641      2474
   age 11-20     0.8418    0.8467    0.8442      1181
   age 21-30     0.8118    0.8326    0.8220      1523
   age 31-40     0.6937    0.6683    0.6808      1010
   age 41-55     0.7106    0.7528    0.7311      1181
   age 56-65     0.6878    0.6646    0.6760       799
   age 66-80     0.7949    0.7596    0.7768       653
    age 80 +     0.9349    0.8343    0.8817       344

    accuracy                         0.8225      9165
   macro avg     0.8046    0.7907    0.7971      9165
weighted avg     0.8226    0.8225    0.8223      9165
```

![download (1).png](https://cdn-uploads.huggingface.co/production/uploads/65bb837dbfb878f46c77de4c/E_8ykSA-ZqEK0_Jtch5dD.png)

---

## Label Space: 8 Classes

```
Class 0: age 01-10  
Class 1: age 11-20  
Class 2: age 21-30  
Class 3: age 31-40  
Class 4: age 41-55  
Class 5: age 56-65  
Class 6: age 66-80  
Class 7: age 80 +
```

---

## Install Dependencies

```bash
pip install -q transformers torch pillow gradio hf_xet
```

---

## Inference Code

```python
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
```

---

## Intended Use

`facial-age-detection` is designed for:

* **Demographic Analytics** – Estimate age distributions in image datasets for research and commercial analysis.
* **Access Control & Verification** – Enforce age-based access in digital or physical environments.
* **Retail & Marketing** – Understand customer demographics in retail spaces through camera-based analytics.
* **Surveillance & Security** – Enhance people classification systems by integrating age detection.
* **Human-Computer Interaction** – Adapt experiences and interfaces based on user age.
