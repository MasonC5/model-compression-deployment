# Model Compression and Deployment for CIFAR-10 Classifier

This project demonstrates how to build, compress, and deploy a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset. It includes model training, compression using knowledge distillation, and a web-based deployment using FastAPI. You can upload a picture and it will decipher the closest option between: Airplane, Bird, Car, Cat, Deer, Dog, Frog, Horse, Ship, or Truck.

## Features

- **CNN Classifier**: Trained a baseline CNN model on CIFAR-10
- **Model Compression**: Applied knowledge distillation to compress the model
- **Deployment**: Integrated FastAPI backend and HTML frontend for serving predictions
- **Benchmarking**: Compared accuracy, inference speed, and file size
- **Downloadable Results**: Save predictions and inference times through the UI

## Project Structure

model-compression-deployment/
├── data/ # CIFAR-10 dataset (downloaded automatically)
├── notebooks/ # Jupyter notebooks for training and evaluation
├── scripts/
│ ├── train_and_quantize.py # Full training, distillation, benchmarking pipeline
│ ├── model_api.py # FastAPI backend
│ ├── models.py # Model definitions
│ └── templates/
│ └── index.html # Frontend interface
└── README.md

## Results

| Model         | Accuracy (%) | Inference Time (s) | Model Size (MB) |
|---------------|--------------|---------------------|------------------|
| Teacher (CNN) | ~84          | ~10.21              | ~9.80            |
| Student (KD)  | ~70.71       | ~8.64               | ~1.08            |

## How to Run

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
2. **Train/Compress Model**
   ```bash
   python scripts/train_and_quantize.py

3. **Start app**
  ```bash
   uvicorn scripts.model_api:app --reload
  ```

4. **Access Frontend**
Open http://127.0.0.1:8000 in your browser.

## Future Work
- Add support for CIFAR-100 or custom datasets
- Extend compression with pruning and quantization
- Dockerize for easier deployment
