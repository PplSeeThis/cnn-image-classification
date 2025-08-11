# CNN for Image Classification

A convolutional neural network (CNN) built with PyTorch to classify images from the Intel Image Classification dataset. This project demonstrates the process of building, training, and evaluating a CNN for a multi-class image classification task.



## 📜 Project Overview

[cite_start]The main goal of this project is to develop a CNN capable of classifying landscape images into one of several categories (e.g., buildings, forest, sea)[cite: 13]. The model is trained on the [Kaggle Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) dataset.

Key steps included:
* **Data Preprocessing:** Images were resized to 128x128 pixels and normalized.
* [cite_start]**Data Augmentation:** Techniques like random horizontal flips and rotations were applied to the training set to improve model generalization[cite: 29, 55, 57].
* [cite_start]**Model Architecture:** A custom CNN was built with three convolutional layers, each followed by a ReLU activation and a MaxPooling layer[cite: 30].
* [cite_start]**Training & Evaluation:** The model was trained for 10 epochs, achieving a final validation accuracy of over 85%[cite: 31, 215].

## 🛠️ Technologies & Libraries

* [cite_start]**Language:** Python 3.10 [cite: 23]
* **Core Libraries:**
    * [cite_start]PyTorch [cite: 24]
    * [cite_start]Torchvision [cite: 24]
    * [cite_start]Matplotlib [cite: 24]
    * NumPy
* [cite_start]**Environment:** Kaggle Notebooks [cite: 25]

## 📈 Results

The model demonstrated strong performance in classifying the images.
* [cite_start]**Peak Validation Accuracy:** **86.93%** (achieved at epoch 9) [cite: 261]
* [cite_start]**Final Test Accuracy:** **85.07%** [cite: 257]

Below are the graphs showing the training loss and validation accuracy per epoch.

| Training Loss | Validation Accuracy |
| :-----------: | :-----------------: |
| ![Training Loss](https://storage.googleapis.com/agent-tools-public-bucket/hosted_tools_images/c234a413-a442-4917-a006-2187f4c5e571.png) | ![Validation Accuracy](https://storage.googleapis.com/agent-tools-public-bucket/hosted_tools_images/e8c85777-6f0e-473d-82d2-8b010c7bd32a.png) |

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/cnn-image-classification.git](https://github.com/your-username/cnn-image-classification.git)
    cd cnn-image-classification
    ```
2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Run the main script:**
    ```bash
    python src/main.py
    ```
