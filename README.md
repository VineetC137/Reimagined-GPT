# 🚀 GPT-Reimagined: KANs vs MLPs

> Exploring Kolmogorov-Arnold Networks as a drop-in replacement for MLPs inside GPT architectures.

This repository implements a **Generative Pre-trained Transformer (GPT)** model using two alternative feed-forward architectures:

- Traditional **Multilayer Perceptrons (MLPs)**
- Novel **Kolmogorov-Arnold Networks (KANs)**

We experimentally evaluate whether **KAN layers can outperform classical MLP blocks** in convergence, expressiveness, and efficiency for language modeling.

---

## 🧠 Why KANs?

The **Kolmogorov-Arnold Representation Theorem** states that any multivariate continuous function can be decomposed into a sum of univariate functions.

KANs exploit this principle to build neural networks that are:

- More interpretable  
- Parameter efficient  
- Expressive with fewer hidden units  

This project investigates whether **KANs can replace MLPs inside GPT architectures without sacrificing language generation quality.**

---

## 🏗️ KAN-GPT Architecture

<div align="center">
  <img src="./images/kan-gpt.png" width="700"/>
</div>

---

## ⚙️ Tech Stack

| Category | Tools |
|--------|------|
| Language | Python |
| Framework | PyTorch |
| Libraries | NumPy, Pandas, SciPy, tqdm, tiktoken |
| Visualization | Matplotlib, TensorBoard |
| Datasets | Tiny Shakespeare, WikiText-2 |
| Tools | Git, Google Colab, Kaggle |

---

## 🎯 Objectives

- Implement GPT using traditional MLP blocks  
- Implement GPT using Kolmogorov-Arnold Networks  
- Compare both approaches on:
  - Training convergence  
  - Validation loss & perplexity  
  - Text generation quality  
  - Parameter efficiency  
  - Resource utilization  

---

# File Directory

<pre><code>
GPT-Reimagined/
├── data/                          # Dataset (tiny Shakespeare data used here)
│   ├── tinyshakespeare/
│   │   ├── input.txt              # Encoded input data
│   │   ├── train.bin              # Encoded training data
│   │   ├── val.bin                # Encoded validation data
├── models/                        # Directory for saving trained models
├── logs/                          # Training logs for TensorBoard
├── archive_logs/                  # Archive of zipped logs
├── main.py                        # Main script to initiate training
├── dataset_shakespeare.py         # Data processing and loading script
├── model_kan.py                   # Kolmogorov-Arnold Network (KAN) model
├── model_mlp.py                   # MLP-based GPT model
├── train.py                       # Training loop for the models
├── config.py                      # Configuration for hyperparameters and paths
├── generate.py                    # Script for generating text with the trained model
├── utils.py                       # Utility functions
├── requirements.txt               # Required dependencies
└── README.md                      # This README file
</code></pre>

## 🛠️ Installation

```bash
git clone https://github.com/your-username/GPT-Reimagined.git
cd GPT-Reimagined
pip install -r requirements.txt
Dataset is automatically downloaded and tokenized using dataset_shakespeare.py.

▶️ Training
bash
Copy code
python main.py
View logs using:

bash
Copy code
tensorboard --logdir=logs
✍️ Text Generation
bash
Copy code
python generate.py
Control output length using max_new_tokens in generate.py.

🔬 Experiment Configuration
Parameter	Value
Block Size	64
Batch Size	64
Learning Rate	2e-5
Epochs	6
Loss	Cross-Entropy
Metric	Validation Loss, Perplexity

📊 Results
KAN-GPT Generated Sample


KAN-based GPT shows:

Faster early convergence

Better coherence at low epochs

Reduced overfitting on small datasets

🧪 Mini Projects
MNIST Classification (MLP vs KAN)

FashionMNIST CNN Classifier

NameGPT

Masked Language Model

Transformer-based Translator

👥 Contributors
Vineet Unde

Shraddha Bhadane
