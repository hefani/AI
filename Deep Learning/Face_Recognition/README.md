# Face Recognition with Triplet Loss
 
This project builds a face recognition system using a Siamese network trained with Triplet Loss. The model learns to embed faces into a feature space where images of the same person are close together and images of different people are far apart.

I included more explanation at the end about my attempts to improve the model.

---

# Overview

- Task: Face Recognition / Verification
- Model: Custom CNN with embedding size = 128
- Loss Function: Triplet Margin Loss
- Dataset: LFW (Labeled Faces in the Wild) from Kaggle
- Framework: PyTorch + facenet-pytorch (for face alignment with MTCNN)
- Goal: Build embeddings for faces and distinguish identities with high accuracy

---

# DataSet

LFW Dataset (Labeled Faces in the Wild)

---

# How It Works

0. Preprocessing
- Detect and crop faces using MTCNN
- Split into train (80%), dev (10%), and test (10%)
- Data augmentation on training (random flips, rotations, color jitter)

1. Face Detection & Alignment
- Uses MTCNN to detect and crop faces before training

2. Dataset Creation
- Generates triplets: (Anchor, Positive, Negative)
- Anchor & Positive are the Same person
- Negative is a Different person

3. Training
- Optimizer: Adam (lr = 0.001)
- Loss: Triplet Margin Loss (margin = 1.0)

4. Evaluation
- Uses pairwise distance between embeddings
- Reports accuracy on validation set

---

# Concepts Used

- Triplet Loss for metric learning
- Data Augmentation (flip, rotation, color jitter)
- Face Detection & Alignment with MTCNN
- PyTorch Datasets & DataLoader for triplets
- GPU Acceleration (CUDA) support
- Embedding-based Face Verification

---

# Files

- `training_face_recognition.ipynb` – for train on data (you can train it on kaggle or google colab)
- `training_face.ipynb` – for run on you computer
- `README.md` – Project documentation
- `face_recognition_model.pth` – Trained model weights for first attempt with 100 epochs

---

# How to Run

1. Install requirements:
   ```bash
   pip install numpy pandas torch torchvision facenet-pytorch matplotlib tqdm
   
2. install dataset from kaggle :
   kaggle datasets download -d jessicali9530/lfw-dataset

3. Run:
   jupyter notebook training_face_recognition.ipynbistance
   jupyter notebook face_recognition.ipynb

---

# Attempts

1. at first I didn't use tau that I compare image's distance with tau, in triplet I compared the positive and negative images with anchor that positive image's distance must be shorter than the negative one.
- with 10 epoch my dev error was 32.55%
- with 50 epoch my dev error was 20.16%
- with 100 epoch my dev error was 14.83 
and I put the first attempt's trained model weights in face_recognition_model.pth
