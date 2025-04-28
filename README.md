# CNN-Fashion-MNIST-Classifier

## 📋 Description
This project implements a **Convolutional Neural Network (CNN)** using **PyTorch** to classify images from the **Fashion-MNIST** dataset.  
It covers data loading, model building, training, evaluation, and visualization of misclassified examples.

---

## 📈 Project Workflow

### 1. Data Loading and Exploration
- Downloaded the Fashion-MNIST training and testing datasets using `torchvision.datasets`.
- Explored dataset dimensions, maximum pixel values, and label classes.

### 2. Model Architecture
- Built a **CNN model** with:
  - Four convolutional layers followed by ReLU activations and MaxPooling.
  - Fully connected layers with Dropout for regularization.
- Output layer predicts one of the 10 Fashion-MNIST classes.

### 3. Training
- Used **CrossEntropyLoss** as the loss function.
- Optimized using the **Adam** optimizer with a learning rate of `0.001`.
- Trained for **10 epochs** using a batch size of **128**.

### 4. Evaluation
- Calculated the **Test Loss** and **Accuracy** on the test dataset.
- Achieved approximately **92.9% accuracy** on the test set.

### 5. Visualization
- Displayed **examples of misclassified images** with predicted and true labels for deeper insight into model errors.

---

## 🛠️ Technologies Used
- Python 3.x
- PyTorch (`torch`, `torch.nn`, `torchvision`)
- Matplotlib (`matplotlib.pyplot`)

---

## ⚙️ Setup

1. Install required libraries:
   ```bash
   pip install torch torchvision matplotlib
