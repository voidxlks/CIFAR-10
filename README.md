# CIFAR-10 Image Classification (PyTorch)

This repository contains a Convolutional Neural Network (CNN) built with PyTorch to classify images from the CIFAR-10 dataset into 10 different classes.

---

## 📦 Requirements

* Python 3.x
* PyTorch
* Torchvision
* NumPy
* Matplotlib (optional, for visualization)
* tqdm (optional, for progress bars)

---

## ⚙️ Installation

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 🚀 Run the Model

To train the model, run:

```bash
python train.py
```

Make sure all dependencies are installed before running.

---

## 📁 Folder Structure

Your project should look like this:

```bash
cifar10-image-classification/
│
├── train.py
├── model.py
├── requirements.txt
├── README.md
└── .git/
```

---

## 🧠 Model Details

* Custom CNN designed for 32×32 CIFAR-10 images
* Uses convolutional layers, batch normalization, and dropout
* Outputs predictions for 10 classes

---

## 📊 Dataset

The CIFAR-10 dataset will automatically download when you run the training script using `torchvision`.

---

## ✨ Future Improvements

* Add data augmentation
* Use pretrained models (ResNet, etc.)
* Save and load trained models
* Add accuracy/loss graphs

---

## 🛠️ Notes

* Training may take a few minutes depending on your hardware
* GPU is recommended but not required

---
