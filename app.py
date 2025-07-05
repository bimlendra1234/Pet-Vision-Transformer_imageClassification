import gradio as gr
import torch
from transformers import ViTForImageClassification, AutoImageProcessor

MODEL_DIR = "./vit_cats_dogs_final_model_20250330_231537"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(MODEL_DIR).to(device)
processor = AutoImageProcessor.from_pretrained(MODEL_DIR)

id2label = model.config.id2label

def classify_image(image):
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
    return {id2label[str(i)]: float(probs[i]) for i in range(len(probs))}

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Pet Vision Transformer Classifier",
    description="Upload an image of a cat or dog to classify using a fine-tuned Vision Transformer."
).launch()
