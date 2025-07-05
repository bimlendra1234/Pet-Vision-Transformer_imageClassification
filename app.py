import gradio as gr
import torch
from PIL import Image
from transformers import ViTForImageClassification, AutoImageProcessor
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "google/vit-base-patch16-224"
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2)
model.load_state_dict(torch.load("DogCatClassification.pth", map_location=device))
model.to(device)
model.eval()

processor = AutoImageProcessor.from_pretrained(model_name)
labels = {0: "Cat", 1: "Dog"}

def classify_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
    return {labels[i]: float(probs[i]) for i in range(2)}

gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Pet Vision Transformer Classifier",
    description="Upload an image of a cat or dog to classify using a fine-tuned ViT model."
).launch()
