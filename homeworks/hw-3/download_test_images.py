"""
Downloads one test image per CIFAR-10 class, plus out-of-distribution samples
from MNIST (handwritten digits) and Oxford Flowers102.
"""
import os
from pathlib import Path
from torchvision import datasets

OUT_DIR = Path("test_images")
CIFAR_DIR = OUT_DIR / "cifar10"
OOD_DIR = OUT_DIR / "out_of_distribution"

CIFAR_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                 'dog', 'frog', 'horse', 'ship', 'truck']

def save_cifar10_samples():
    print("Downloading CIFAR-10 test set...")
    CIFAR_DIR.mkdir(parents=True, exist_ok=True)

    dataset = datasets.CIFAR10("./data", train=False, download=True)

    found = {}
    for img, label in dataset:
        cls = CIFAR_CLASSES[label]
        if cls not in found:
            path = CIFAR_DIR / f"{cls}.png"
            img.save(path)
            found[cls] = True
            print(f"  Saved: {path}")
        if len(found) == 10:
            break

def save_ood_samples():
    print("\nDownloading OOD samples (MNIST digits — clearly not CIFAR-10)...")
    OOD_DIR.mkdir(parents=True, exist_ok=True)

    mnist = datasets.MNIST("./data", train=False, download=True)
    ood_labels = {0: "digit_zero", 3: "digit_three", 7: "digit_seven"}
    for img, label in mnist:
        if label in ood_labels:
            path = OOD_DIR / f"{ood_labels[label]}.png"
            img.convert("RGB").save(path)
            print(f"  Saved: {path}")
            del ood_labels[label]
        if not ood_labels:
            break

    print("\nDownloading OOD samples (Oxford Flowers102 — flowers not in CIFAR-10)...")
    try:
        flowers = datasets.Flowers102("./data", split="test", download=True)
        for i, (img, _) in enumerate(flowers):
            if i >= 3:
                break
            path = OOD_DIR / f"flower_{i+1}.jpg"
            img.save(path)
            print(f"  Saved: {path}")
    except Exception as e:
        print(f"  Flowers102 download failed ({e}), skipping.")

if __name__ == "__main__":
    save_cifar10_samples()
    save_ood_samples()

    print(f"\nDone. Images saved to {OUT_DIR}/")
    print("\nTo test inference (replace EXTERNAL_IP):")
    print(f"  EXTERNAL_IP=<your-ip>")
    for cls in CIFAR_CLASSES:
        print(f"  curl -s -X POST http://$EXTERNAL_IP/predict -F image=@{CIFAR_DIR}/{cls}.png | python3 -m json.tool")
    print(f"  # OOD images:")
    for f in ["digit_zero.png", "digit_three.png", "flower_1.jpg"]:
        print(f"  curl -s -X POST http://$EXTERNAL_IP/predict -F image=@{OOD_DIR}/{f} | python3 -m json.tool")
