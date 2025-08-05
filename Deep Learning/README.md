# Cat vs Dog Image Classifier
 
This project builds a binary image classifier that distinguishes between cats and dogs using transfer learning with a pre-trained deep convolutional neural network (ResNet50).

---

# Overview

- Classify images as cat or dog
- Model: ResNet50 (pre-trained on ImageNet)
- Dataset: Kaggle’s Dogs vs. Cats (25,000 labeled images)
- Framework: PyTorch
- Achieves high accuracy using fine-tuned ResNet model

---

# DataSet

Contains features like:
cats and dogs images

The target variable is cat and dog.

---

# How It Works
1. Data Preprocessing
   Resizes and normalizes images
   Augments training data with random flips & rotations

3. Model
   Loads ResNet50 with pre-trained weights
   Freezes early layers, fine-tunes the final layers
   Uses CrossEntropyLoss and Adam optimizer

3. Training
   Trained for several epochs with mini-batches
   Validated on a hold-out dataset

4. Evaluation
   Reports accuracy and loss on validation set
   Can predict on new unseen images

---

# Concepts Used

- Transfer Learning with Pre-trained CNN (ResNet50)
- Convolutional Neural Networks (CNNs)
- Fine-tuning and Freezing Layers
- Image Data Augmentation
- Binary Classification with CrossEntropyLoss
- Mini-batch Gradient Descent using Adam Optimizer
- Train/Validation Split Strategy
- GPU Acceleration (CUDA) with PyTorch
- Evaluation Metrics: Accuracy, Loss
- Data Normalization (ImageNet mean and std)
- PyTorch Dataset & DataLoader APIs

---

# Files

- `cat_vs_dog.ipynb` – Main implementation  
- `README.md` – Project documentation

---

# How to Run

1. Install requirements:
   ```bash
   pip install numpy pandas scikit-learn
   pip install torch torchvision matplotlib numpy
   
2. install dataset from kaggle :
   https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

3. Run:
   jupyter notebook cat_vs_dog_resnet.ipynb

