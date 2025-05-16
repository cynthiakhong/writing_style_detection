# User Guide

This guide will help you get started with the document analysis system for analyzing text documents in English or Chinese.

## Getting Started

### Step 1: Download the Repository

1. Click the green "Code" button
2. Choose "Download ZIP"
3. Extract the ZIP file to a location on your computer

Alternatively, if you have Git installed, you can clone the repository.

### Step 2: Install Requirements

1. Make sure you have Python installed (Python 3.7+ recommended)
2. Open a terminal/command prompt
3. Navigate to the project folder
4. Install all required packages:

   ```
   pip install -r requirements.txt
   ```

### Step 3: Prepare Your Documents

1. Place your `.docx` documents in a folder
2. For English documents: You can organize files in multiple subfolders
3. For Chinese documents: Place all files in a single folder
   > **Note**: The system currently only supports .docx files

### Step 4: Configure Your Analysis

Open `config.py` in any text editor and modify these key settings:

```python

# Set paths to your document folders
BASE_DATA_PATH = "C:/path/to/your/documents"  # Change this to your folder path

# For English documents, specify subfolder names if needed
ENGLISH_SUBDIRS = [
    "folder1",
    "folder2"
]

# For Chinese documents, set the path
CHINESE_DATA_PATH = "C:/path/to/chinese/documents"
# Set the language you want to analyze
LANGUAGE_CHOICE = "eng"  # Use "eng" for English or "cn" for Chinese

# Choose your dimensionality reduction method
AUTOENCODER_CHOICE = "vae"  # Use "vae" or "ae"

```

### Step 5: Run the Analysis

1. Open a terminal/command prompt
2. Navigate to the project folder
   ```
   cd "C:/path/to/your/project folder"
   ```
3. Run the analysis:
   ```
   python main.py
   ```
4. The system will:
   - Load your documents
   - Generate embeddings
   - Process them through autoencoders
   - Perform clustering
   - Create visualizations

### Step 6: View Results

Results are saved in the `result/{language}/{autoencoder_type}/{timestamp}` folder:

- Clustering visualizations (PCA and t-SNE)
- Method comparison charts
- Performance reports

## Customizing Your Analysis

### Change Embedding Models

In `config.py`, you can enable/disable different embedding methods:

```python
# Choose which methods to use
VECTORIZE_METHODS = [
    'transformers',  # BERT, RoBERTa, etc.
    'tfidf',         # Remove comment to use TF-IDF
    'word2vec'       # Remove comment to use Word2Vec
]
```

### Adjust Autoencoder Settings

Modify autoencoder parameters to adjust the dimensionality reduction:

```python
# Autoencoder parameters
AUTOENCODER_PARAMS = {
    "epochs": 100,         # Training epochs (reduce for faster results)
    "batch_size": 64,      # Batch size for training
}
```

### Change Clustering Settings

Adjust how many clusters the system should try:

```python
# Clustering parameters
CLUSTERING_PARAMS = {
    "k_range": (2, 10),    # Min and max number of clusters to try
    "n_init": 10           # Number of times to run clustering
}
```

## Common Issues

1. **Out of memory errors**: Reduce batch size or use fewer documents
2. **Slow processing**: Reduce the number of embedding methods or use simpler models
3. **Installation issues**: Make sure you have the correct Python version and all dependencies
4. **Missing modules**: If you get module not found errors, try installing the specific package:
   ```
   pip install [package_name]
   ```
