# Deep Learning Fast Track Practice (SK hynix & LG Energy Solution)

This repository contains hands-on practice code for deep learning concepts used in the Fast Track Training Program conducted for SK hynix and LG Energy Solution employees. Each module is designed to provide intuitive understanding and practical experience with foundational and advanced deep learning topics.

---

## Practice Modules

### 1. MNIST_MLP
- Basic image classification using a Multi-Layer Perceptron (MLP)
- Dataset: MNIST (handwritten digits)
- Focus: Feedforward neural network structure, activation, loss functions

### 2. Autoencoder
- Unsupervised learning to compress and reconstruct images
- Dataset: MNIST
- Focus: Encoder–decoder structure, latent space visualization

### 3. CIFAR_MLP
- Image classification on CIFAR-10 using MLP
- Dataset: CIFAR-10
- Focus: Limitations of MLP in complex visual tasks

### 4. CIFAR_CNN
- CNN-based image classification
- Dataset: CIFAR-10
- Focus: Convolution layers, pooling, feature extraction

### 5. CIFAR_CNN_Augmentation
- Improved performance through data augmentation
- Dataset: CIFAR-10
- Techniques: Flip, crop, normalize, etc.
- Focus: Generalization and overfitting mitigation

### 6. ResNet
- Deep residual network implementation
- Dataset: CIFAR-10
- Focus: Skip connections, deep model training stability

### 7. pretrained_model
- Using pretrained CNNs (e.g., VGG, ResNet)
- Dataset: CIFAR-10 or custom dataset
- Focus: Feature reuse, inference without full training

### 8. transfer_learning
- Fine-tuning pretrained models on new datasets
- Focus: Efficient training on small or domain-specific datasets

---

## Environment

- Python ≥ 3.8  
- PyTorch ≥ 1.10 / TensorFlow (for selected modules)  
- Torchvision, Matplotlib, Numpy, etc.

Install dependencies using:

```bash
pip install -r requirements.txt

