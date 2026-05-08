import io
import torch
import torch.nn as nn
from torchvision import transforms, models
from flask import Flask, request, jsonify
from PIL import Image

CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
MODEL_PATH = "/mnt/model/resnet18.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model at startup
model = models.resnet18()
model.fc = nn.Linear(512, 10)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()
print(f"Model loaded from {MODEL_PATH} on {DEVICE}")

val_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image_bytes = request.files["image"].read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = val_transforms(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        predicted_idx = outputs.argmax(dim=1).item()

    return jsonify({"prediction": CLASSES[predicted_idx]})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
