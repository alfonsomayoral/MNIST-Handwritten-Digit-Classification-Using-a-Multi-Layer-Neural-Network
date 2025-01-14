# **MNIST Handwritten Digit Classification using a Multi-Layer Neural Network**
This project demonstrates the implementation of a two-layer neural network for classifying handwritten digits from the MNIST dataset

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)  
[![Libraries](https://img.shields.io/badge/Libraries-Scikit--learn%2C%20Matplotlib%2C%20Seaborn%2C%20Numpy-green)](https://scikit-learn.org/)  
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)](#)

## **📖 Overview**
This project demonstrates the implementation of a two-layer neural network for classifying handwritten digits from the MNIST dataset. By leveraging foundational concepts in machine learning and neural networks, the notebook offers an end-to-end implementation, including:

- Preprocessing and visualizing MNIST data.
- Designing and training a simple neural network from scratch.
- Evaluating model performance through accuracy, confusion matrices, and ROC curves.
  
The primary goal is to showcase a clean, modular approach to multiclass classification using neural networks, providing a foundation for further experimentation and extension.

---

## **📋 Table of Contents**
- 🚀 [Introduction](#-introduction)
- 🛠  [Implementation Details & Structure](#-implementation-details--structure)
- 📌 [How to Use](#-how-to-use)
- 📊 [Visual Results](#-visual-results)

---

### 🚀 **Introduction**

The MNIST dataset, a widely recognized benchmark for handwritten digit classification, consists of grayscale images of digits (0–9). Each image is 28x28 pixels, and the dataset includes:

- **60,000 training images**
- **10,000 test images**
  
This project implements a two-layer neural network architecture:
- **Input Layer**: 784 nodes (28x28 pixel values flattened).
- **Hidden Layer**: 256 nodes with a sigmoid activation function.
- **Output Layer**: 10 nodes with a softmax activation function.
  
Key steps include:
- Preprocessing: Vectorizing images and normalizing pixel values.
- Training: Using cross-entropy loss and backpropagation.
- Evaluation: Assessing accuracy, visualizing misclassified examples, and plotting confusion matrices and ROC curves.

---

## 🛠️ **Implementation Details**

#### **1. Data Preprocessing**
- **Dataset**: The MNIST dataset was loaded using the `Keras` library.
- **Reshaping**: Images were reshaped into vectors of size 784 (28x28 flattened).
- **Normalization**: Pixel values were normalized to the range [0, 1].
- **Balanced Classes**: Equal representation of all digit classes was ensured by selecting a fixed number of samples per class.

#### **2. Neural Network Architecture**
- **Layers**:
  - **Input Layer**: 784 nodes, corresponding to the pixel values.
  - **Hidden Layer**: 256 nodes with a **sigmoid activation function**.
  - **Output Layer**: 10 nodes with a **softmax activation function**.
- **Weight Initialization**: Random initialization with a small standard deviation (mean = 0, std = 0.01).
- **Loss Function**: Multi-class cross-entropy loss to optimize classification probabilities.
- **Optimizer**: Gradient Descent to update weights and biases.

#### **3. Training**
- **Forward Propagation**:
  - Computes the activations for each layer.
  - Uses softmax to generate class probabilities in the output layer.
- **Backward Propagation**:
  - Calculates gradients for weights and biases using the chain rule.
  - Applies a numerically stable implementation for softmax and loss derivatives.
- **Optimization**:
  - Weights and biases updated iteratively using gradient descent.
  - Trained over 100 epochs with a learning rate of 0.00001.

#### **4. Evaluation**
- **Accuracy**:
  - Evaluated on both training and test datasets.
  - Compares predicted labels with ground truth to calculate the percentage of correct predictions.
- **Visualizations**:
  - **Confusion Matrix**: Shows the distribution of correct and incorrect predictions for each class.
  - **ROC Curves**: Plots True Positive Rate (TPR) vs. False Positive Rate (FPR) for each class, providing a visual and numerical assessment of the model’s performance.
- **Qualitative Analysis**:
  - Displays correctly and incorrectly classified images from both training and test sets.
  - Highlights true positives, true negatives, false positives, and false negatives.
 
  ---

  ### 📌 **How to Use**

Follow these steps to set up and run the project locally:

#### **1. Clone the Repository**
Clone the repository and navigate to the project directory:
```bash
git clone <repository_url>
cd project_directory
```
#### 2. **Install Required Dependencies**
- Make sure you have Python 3.7 or later installed. Install the necessary libraries by running:
   ```bash
   pip install -r requirements.txt
- This will install dependencies such as: NumPy for numerical computations, Scikit-learn for implementing K-means and PCA, Matplotlib and Seaborn for data visualization, scikit-image for image processing.

#### 3. Launch the Jupyter Notebook
- Run the following command to open the Jupyter Notebook interface:
   ```bash
   jupyter notebook Multilayer_Neural_Network_Classification_Handwritten_Digits.ipynb
- This will open the notebook in your default web browser.

#### 4. Execute the Notebook
Run the notebook cells sequentially to:
- Preprocess the MNIST dataset.
- Train the two-layer neural network.
- Evaluate the model’s performance.
- Visualize results such as loss, confusion matrices, and ROC curves.
  
#### 5. Experiment with Parameters
Customize and experiment with the model by modifying:
- The number of nodes in the hidden layer.
- Learning rate and the number of epochs.
- Training sample sizes for each class.

#### 6. Save Results
- Save plots or outputs by running:
```bash
plt.savefig('output_name.png', dpi=300)
```

#### 7. Troubleshooting
- Ensure Python 3.7+ is installed.
If a library is missing, install it manually:
```bash
pip install <library_name>
```
- Verify that the notebook file (mnist_two_layer_nn.ipynb) and dataset paths are correctly set up.

**With these steps, you’re all set to explore and experiment with the project! 🚀**

---
## 📊 **Results and Visualizations**
1. Training and Test Accuracy
- Training Accuracy: ~88.5%
- Test Accuracy: ~86.8%
  
2. Loss Over Epochs
The loss consistently decreases, demonstrating successful training convergence.


3. Confusion Matrix
The confusion matrix highlights the classifier's performance across all classes.

Training Set Confusion Matrix

Test Set Confusion Matrix

4. ROC Curves and AUC
ROC curves visualize the trade-off between the True Positive Rate (TPR) and False Positive Rate (FPR) for each class. The Area Under the Curve (AUC) quantifies classification performance.


5. Correct and Misclassified Examples
Examples of correctly classified and misclassified images from both training and test sets provide qualitative insights into the model's predictions.

**Training Set**

**Test Set**

---

## 📈 **Future Work**
Extend the architecture to a deeper neural network or Convolutional Neural Network (CNN).
Experiment with different activation functions and optimizers.
Apply data augmentation techniques to improve generalization.
