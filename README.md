# Document Analysis System

A framework for processing, embedding, and clustering text documents in both English and Chinese using transformer models, autoencoders, and unsupervised learning techniques.

## Overview

This system provides a pipeline for document analysis with the following capabilities:

- Processing documents in both English and Chinese
- Generating embeddings using transformer models, TF-IDF, and Word2Vec
- Reducing dimensionality through autoencoders (AE and VAE)
- Clustering documents using K-means and Gaussian Mixture Models
- Visualizing results with PCA and t-SNE

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
torch
numpy
pandas
transformers
tqdm
scikit-learn
gensim
matplotlib
seaborn
nltk
jieba
python-docx
```

## Project Structure

```
your_directory/
├── main.py                # Main execution script
├── config.py              # Configuration settings
├── process_data.py        # Document loading and preprocessing
├── vectorize.py           # Embedding generation methods
├── autoencoders.py        # Autoencoder models for dimensionality reduction
├── utils.py               # Clustering and visualization utilities
├── Results/               # Analysis results and visualizations
│   ├── English/           # Results for English documents
│   │   ├── AE/            # Standard autoencoder results
│   │   └── VAE/           # Variational autoencoder results
│   └── Chinese/           # Results for Chinese documents
│       ├── AE/            # Standard autoencoder results
│       └── VAE/           # Variational autoencoder results
└── requirements.txt       # Package dependencies
```

## Usage

1. Configure your analysis in `config.py`:

   - Set document paths
   - Choose language (English or Chinese)
   - Select embedding models
   - Configure autoencoder parameters

2. Run the analysis:

   ```bash
   python main.py
   ```

3. Results and visualizations will be saved in the `result/` directory.

## Key Features

### Embedding Methods

- **Transformer Models**: BERT, RoBERTa, DistilBERT, Sentence-BERT, XLM
- **Traditional Methods**: TF-IDF, Word2Vec
- **Hybrid Models**: Combinations of transformer embeddings

### Dimensionality Reduction

- **Standard Autoencoder (AE)**: Optimized for text embeddings
- **Variational Autoencoder (VAE)**: Probabilistic encoder with KL divergence regularization

### Analysis

- K-means and GMM clustering with automatic selection of optimal cluster count
- PCA and t-SNE visualizations
- Comparative analysis across embedding methods

## Configuration

The system is configured through `config.py`. Key settings include:

```python
# Language choice
LANGUAGE_CHOICE = "eng"  # Options: "eng" or "cn"

# Autoencoder choice
AUTOENCODER_CHOICE = "vae"  # Options: "ae" or "vae"

# Vectorization methods
VECTORIZE_METHODS = [
    'transformers',  # Expands to all transformer models
    'tfidf',
    'word2vec'
]

# Autoencoder parameters
AUTOENCODER_PARAMS = {
    "epochs": 100,
    "batch_size": 64,
    "encoding_dim": 64
}

# Clustering parameters
CLUSTERING_PARAMS = {
    "k_range": (2, 12),  # Min and max clusters to try
    "n_init": 10
}
```

## Results

The system was evaluated on both English and Chinese documents using two types of autoencoders (AE and VAE):

### English Documents

- [English with Standard Autoencoder (AE)](./Results/English/AE)
- [English with Variational Autoencoder (VAE)](./Results/English/VAE)

### Chinese Documents

- [Chinese with Standard Autoencoder (AE)](./Results/Chinese/AE)
- [Chinese with Variational Autoencoder (VAE)](./Results/Chinese/VAE)

Key findings:

- Transformer-based models outperformed traditional methods in all scenarios
- VAE generally produced better clustering than standard AE
- For English, SBERT achieved the highest silhouette score (0.7099)
- For Chinese, roberta-zh performed best with a score of (0.3127)
