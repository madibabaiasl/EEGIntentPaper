# EEG-Based Intent Recognition and Robotic Control using Transformer Networks


## 🧠 Project Overview

This project implements a real-time brain-robot interface (BRI) that uses EEG signals to classify user motor intent and control the **Kinova Gen3** robotic arm. A transformer-based deep learning model, **TransNN-MHA**, is proposed to recognize real vs. imaginary motor intentions with high accuracy, enabling robotic tasks such as grasping and placing objects without hardcoded commands.

The system leverages temporal dependencies in EEG signals to decode intent and execute motor commands on the robot in real-time.





## ⚙️ Key Features

- 🧠 **Real vs. Imagined EEG Motor Intent Classification**: Accurately distinguishes between real and imagined motor tasks using EEG signals, vital for users with motor impairments.

- 🔁 **Transformer-Based Architecture**: Employs a lightweight Transformer with **Multi-Head Attention (MHA)** to capture both local and global temporal features in EEG data.

- 🧹 **Comprehensive Preprocessing Pipeline**: Uses ICA, bandpass and notch filters, z-score normalization, and spectral feature extraction to clean EEG data.

- 📊 **Handcrafted Features + SHAP Interpretability**: Extracts 70 domain-specific features with SHAP-based explanation to highlight feature importance.

- ⚖️ **SMOTE for Data Imbalance**: Balances classes in the training set using synthetic oversampling while keeping the test data untouched.

- 🔍 **Subject-Independent 6-Fold Cross-Validation**: Ensures robust model performance across subjects by using separate individuals for training and testing.

- ⚡ **High Accuracy and Efficiency**: Achieves **92.39% accuracy ±1.34%**, outperforming CNN-Transformer, GRU-Transformer, and various ML baselines.

- 🛠️ **Minimalist Design for Real-Time Use**: Simplifies Transformer without positional encodings or decoders, making it suitable for embedded deployment.

- 🤖 **Assistive Robotics Focus**: Tailored for use in BCI-driven assistive systems for prosthetics and neurorehabilitation.

---

## 📊 Dataset

The dataset used is **publicly available from PhysioNet**:

> EEG Motor Movement/Imagery Dataset  
> https://physionet.org/content/eegmmidb/1.0.0/

- Subjects: 109 participants  
- Channels: 64-channel EEG  
- Tasks: Real and imaginary motor movements (left hand, right hand, both hands, feet)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your_username/eeg-robot-transformer.git
cd eeg-robot-transformer

\section*{Citation}
@article{yourpaper2025transnn,
  title={TransNN-MHA: A Transformer-Based Model to Distinguish Real and Imaginary Motor Intent for Assistive Robotics},
  author={Your Name and Collaborators},
  journal={IEEE Access},
  year={2025}
}


