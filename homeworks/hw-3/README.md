# Homework 3 — DL Training & Inference on GKE

COMS 6998E — Applied Machine Learning in the Cloud

## Overview

This project trains a ResNet18 image classifier on CIFAR-10 using a Kubernetes Job on GKE, saves the model to a PersistentVolumeClaim, and serves inference via a Flask web server exposed through a Kubernetes LoadBalancer Service.

## Architecture

```
[Training]                              [Inference]
training/train.py                       inference/app.py
training/Dockerfile                     inference/Dockerfile
    |                                       |
    v                                       v
gcr.io/$PROJECT_ID/resnet18-trainer    gcr.io/$PROJECT_ID/resnet18-inference
    |                                       |
    v                                       v
k8s/train-job.yaml (Job)           k8s/inference-deployment.yaml (Deployment)
    |                                       |
    +-----> k8s/pvc.yaml (PVC) <-----------+
            /mnt/model/resnet18.pth
                                    k8s/inference-service.yaml (LoadBalancer)
                                            |
                                            v
                                    http://<EXTERNAL-IP>/predict
```

## File Structure

```
hw-3/
├── training/
│   ├── train.py                    # ResNet18 training script (CIFAR-10)
│   └── Dockerfile
├── inference/
│   ├── app.py                      # Flask inference server
│   └── Dockerfile
├── k8s/
│   ├── pvc.yaml                    # PersistentVolumeClaim for model storage
│   ├── train-job.yaml              # Kubernetes Job for training
│   ├── inference-deployment.yaml   # Kubernetes Deployment for inference
│   └── inference-service.yaml      # Kubernetes LoadBalancer Service
├── download_test_images.py         # Downloads CIFAR-10 + OOD test images
├── test_inference.sh               # Runs all test images against /predict endpoint
└── requirements.txt                # Python dependencies for local scripts
```

## Dependencies

### Local machine

- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install)
- `kubectl` — installed via: `gcloud components install kubectl`

No local Docker installation is required — images are built using **Google Cloud Build**.

### GCP services used

- Google Kubernetes Engine (GKE)
- Google Container Registry (GCR)
- Cloud Build
- Persistent Disk (via PVC)

## GCP Setup

### 0. Set your project ID and zone

All commands below use `$PROJECT_ID` and `$ZONE`. Set them once before running anything:

```bash
export PROJECT_ID=your-gcp-project-id
export ZONE=us-central1-a   # change to whichever zone has GPU availability
gcloud config set project $PROJECT_ID
```

### 1. Enable required APIs

```bash
gcloud services enable container.googleapis.com cloudbuild.googleapis.com compute.googleapis.com
```

### 2. Create a dedicated GKE node service account

Google recommends using a custom service account for GKE nodes rather than the default Compute Engine SA, which may not be provisioned on new projects.

```bash
gcloud iam service-accounts create gke-node-sa \
  --display-name="GKE Node Service Account" \
  --project=$PROJECT_ID

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/editor"
```

> **Note:** `roles/editor` is used here for simplicity in a dev/homework environment. It covers image pulls from GCR (backed by Artifact Registry in projects created after 2023), logging, and monitoring. In production, use least-privilege roles instead.

### 3. Create GKE cluster with T4 GPU node pool

**GPU cluster (recommended):**

```bash
gcloud container clusters create amlc-cluster \
  --zone $ZONE \
  --num-nodes 1 \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1 \
  --disk-size=50GB \
  --service-account=gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

> **CPU-only fallback:** If no GPU zone is available (global stockout), drop the `--accelerator` flag. Training will take longer (~1–2 hrs vs ~15 min) but is fully supported by the code.
>
> ```bash
> gcloud container clusters create amlc-cluster \
>   --zone $ZONE \
>   --num-nodes 1 \
>   --machine-type n1-standard-4 \
>   --disk-size=50GB \
>   --service-account=gke-node-sa@${PROJECT_ID}.iam.gserviceaccount.com
> ```

### 4. Connect kubectl to the cluster

```bash
gcloud container clusters get-credentials amlc-cluster --zone $ZONE
```

### 5. Install NVIDIA GPU drivers on the cluster nodes

```bash
kubectl apply -f https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/cos/daemonset-preloaded.yaml
```

## Build & Push Docker Images (via Cloud Build — no local Docker needed)

### Training image

```bash
cd training/
gcloud builds submit --tag gcr.io/$PROJECT_ID/resnet18-trainer:latest .
cd ..
```

### Inference image

```bash
cd inference/
gcloud builds submit --tag gcr.io/$PROJECT_ID/resnet18-inference:latest .
cd ..
```

## Deploy to GKE

The k8s YAML files reference `$PROJECT_ID` via `envsubst` (a standard GNU utility — no extra install needed). Make sure the env var is still set from the setup step, then apply:

### 1. Create the PersistentVolumeClaim

```bash
kubectl apply -f k8s/pvc.yaml
kubectl get pvc
```

### 2. Run the training Job

```bash
envsubst < k8s/train-job.yaml | kubectl apply -f -
kubectl get jobs
kubectl logs -f job/resnet18-training        # stream training logs in real time
```

Training takes ~15 minutes on a T4 GPU. To block until the job finishes:

```bash
kubectl wait --for=condition=complete job/resnet18-training --timeout=30m
```

The job completes when the model is saved to `/mnt/model/resnet18.pth` on the PVC.

### 3. Deploy the inference server

```bash
envsubst < k8s/inference-deployment.yaml | kubectl apply -f -
kubectl apply -f k8s/inference-service.yaml
```

### 4. Get the external IP

```bash
kubectl get service resnet18-inference-svc
```

Wait until `EXTERNAL-IP` is assigned (may take 1–2 minutes).

## Running Inference

### Download test images

A helper script downloads one image per CIFAR-10 class from the test set, plus out-of-distribution samples (MNIST digits and Oxford Flowers102):

```bash
pip install -r requirements.txt   # torch, torchvision, pillow
python download_test_images.py
```

This creates:

```
test_images/
├── cifar10/          # airplane.png, automobile.png, ... truck.png
└── out_of_distribution/   # digit_zero.png, digit_three.png, flower_1.jpg, ...
```

### Run inference tests

```bash
./test_inference.sh <EXTERNAL-IP>
```

Expected output:

```
=== CIFAR-10 (in-distribution) ===
airplane                            -> airplane
automobile                          -> automobile
bird                                -> bird
cat                                 -> cat
deer                                -> deer
dog                                 -> dog
frog                                -> frog
horse                               -> horse
ship                                -> ship
truck                               -> truck

=== Out-of-distribution ===
flower_1                            -> frog
flower_2                            -> frog
flower_3                            -> frog
digit_zero (MNIST)                  -> frog
digit_three (MNIST)                 -> horse
digit_seven (MNIST)                 -> bird
```

The model achieves **100% accuracy on all 10 in-distribution classes**. OOD images (flowers, handwritten digits) are force-mapped to the nearest CIFAR-10 class — mostly animals like frog, horse, and bird, which have similar blob-like shapes. This is expected behavior of a closed-set softmax classifier: it has no "unknown" option and must always pick one of the 10 classes.

### Manual curl

```bash
curl -X POST http://<EXTERNAL-IP>/predict \
  -F "image=@test_images/cifar10/cat.png"
```

Expected response:

```json
{ "prediction": "cat" }
```

Health check:

```bash
curl http://<EXTERNAL-IP>/health
```

## Cleanup (to avoid GCP charges)

```bash
kubectl delete -f k8s/
gcloud container clusters delete amlc-cluster --zone $ZONE
```

## Model Details

| Property        | Value                                        |
| --------------- | -------------------------------------------- |
| Architecture    | ResNet18 (pretrained on ImageNet)            |
| Dataset         | CIFAR-10 (10 classes, downloaded at runtime) |
| Final layer     | `nn.Linear(512, 10)`                         |
| Training epochs | 10                                           |
| Batch size      | 64                                           |
| Optimizer       | Adam (lr=0.001)                              |
| GPU             | NVIDIA Tesla T4                              |

## CIFAR-10 Classes

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`
