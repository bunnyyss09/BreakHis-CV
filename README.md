# Breast Cancer Histopathology Classifier (BreakHis)

## Overview
This project aims to classify breast tumor histopathology images using deep learning. It leverages state-of-the-art convolutional neural networks (CNNs) to distinguish between benign and malignant tumors, automating a process that is traditionally performed by pathologists.

## Dataset: BreaKHis
The **Breast Cancer Histopathological Image Classification (BreakHis)** dataset consists of **9,109 microscopic images** collected from **82 patients** at various magnification levels (**40X, 100X, 200X, and 400X**). The dataset contains:
- **2,480 benign** tumor samples
- **5,429 malignant** tumor samples
- Images are in **700x460 pixels**, **RGB format**, **8-bit depth per channel**

More details: [BreakHis Dataset](https://www.kaggle.com/datasets/ambarish/breakhis)

## Project Highlights
- **Implemented multiple models**: Custom CNN, EfficientNet, DenseNet169, and ResNet50.
- **Preprocessed and augmented** the dataset to enhance model generalization.
- **Used evaluation metrics**: Accuracy, Precision, Recall, and F1-score for performance analysis.
- **Automating histopathology classification**: Reducing time and manual effort required by pathologists.

## Results
- Achieved **high classification accuracy** with deep learning models.
- Comparative analysis of models for performance insights.
- Model predictions help **automate histopathology classification**, reducing manual effort.

## Future Improvements
- Implementing **Vision Transformers (ViTs)** for improved classification.
- Fine-tuning pre-trained models with advanced techniques.
- Exploring **GAN-based augmentation** for synthetic data generation.

## Acknowledgments
- Dataset: **BreakHis (Parana, Brazil)**
- Libraries used: **PyTorch, TensorFlow, WandB, OpenCV**


