# Tumor-Detection

Tumor Detection Using InceptionResNetV2
Project Overview
This project focuses on detecting tumors in medical images using a convolutional neural network (CNN) based on the InceptionResNetV2 architecture. The model leverages transfer learning by using a pre-trained InceptionResNetV2 model, which has been fine-tuned on a custom dataset to classify images as either containing a tumor or not.

Model Architecture
Pre-trained Model:
InceptionResNetV2 is utilized as the base model, pre-trained on the ImageNet dataset. The model is used for feature extraction by excluding the top layers (include_top=False).
Custom Layers:
Global Average Pooling Layer: Reduces the spatial dimensions of the feature maps from the base model.
Dense Layers: Two fully connected layers with 512 units each and ReLU activation functions, along with dropout layers (dropout rate of 0.3) to reduce the risk of overfitting.
Output Layer: A dense layer with 2 units and softmax activation is added to perform binary classification, predicting whether the image contains a tumor or not.
Model Compilation:
The model is compiled using the Stochastic Gradient Descent (SGD) optimizer, with a learning rate of 0.01, momentum of 0.9, and Nesterov acceleration. The loss function used is categorical crossentropy, and accuracy is used as the evaluation metric.
Training Strategy:
Layers up to the 780th layer of the InceptionResNetV2 model are frozen to retain the features learned from the ImageNet dataset. The remaining layers are fine-tuned on the tumor detection dataset.
Installation and Setup
Prerequisites
Python 3.x
TensorFlow 2.x
NumPy
Pandas
Matplotlib
Installation
To install the necessary packages, run:

bash
Copy code
pip install tensorflow numpy pandas matplotlib
Dataset
Prepare your dataset, ensuring it is organized into appropriate directories, such as train/tumor, train/no_tumor, validation/tumor, and validation/no_tumor.
Ensure the dataset is preprocessed and ready for training.
Running the Model
Clone the repository:
bash
Copy code
git clone <repository_url>
cd <repository_directory>
Prepare your dataset in the required format.
Modify the script to load your dataset.
Run the training script:
bash
Copy code
python tumor_detection.py
