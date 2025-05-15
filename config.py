"""
Configuration file for document analysis system.
Contains language settings, paths, and embedding model configurations.
"""
# =====================================================================
# PATH CONFIGURATIONS
# =====================================================================

# Base data path
BASE_DATA_PATH = r'C:\Users\cynth\OneDrive - Nanyang Technological University\nlp\OneDrive_2025-02-18'

# Subdirectories for English documents
ENGLISH_SUBDIRS = [
    r'VAE WRITING STYLE\asgm 1 simulation',
    r'VAE WRITING STYLE\asgm 2 simulation',
    r'VAE WRITING STYLE\asgm 3 simulation',
    r'VAE WRITING STYLE\task 6 simulation',
    r'Practice - Singapore GE',
    r'Simulation'
]

# Chinese documents path
CHINESE_DATA_PATH = r'C:\Users\cynth\OneDrive - Nanyang Technological University\nlp\OneDrive_2025-02-18\Simulation'

# =====================================================================
# ANALYSIS CONFIGURATIONS
# =====================================================================

# Language configurations
LANGUAGE_CHOICE = "eng"  # Options: "eng" or "cn"

# Autoencoder choice
AUTOENCODER_CHOICE = "ae"  # Options: "ae" or "vae"

# Vectorization methods to use (will be expanded in main.py to include specific transformer models)
# You can disable specific method groups by commenting them out
VECTORIZE_METHODS = [
    'transformers',  # Will expand to all transformer models defined below
   # 'tfidf',
    #'word2vec'
]

# =====================================================================
# VECTORIZATION CONFIGURATIONS
# =====================================================================

# TF-IDF configuration
TFIDF_CONFIG = {
    'max_features': 10000,
    'tokenizer': None  # Can be set to a custom tokenizer if needed
}

# Word2Vec configuration
WORD2VEC_CONFIG = {
    'model_name': 'word2vec-google-news-300',
    'custom_model': None  # Can be set to a custom pre-trained model if needed
}

# Embedding configurations for Chinese
CHINESE_EMBEDDING_CONFIGS = [
    {
        "name": "cbert", 
        "language": "chinese", 
        "args": {
            "model_type": "chinese-bert",
            "pooling": "cls"  # Use CLS token pooling
        }
    },
    {
        "name": "sbert", 
        "language": "chinese", 
        "args": {
            "model_type": "sentence-bert",
            "pooling": "mean"  # Use mean pooling for Sentence BERT
        }
    },
    {
        "name": "roberta-zh", 
        "language": "chinese", 
        "args": {
            "model_type": "roberta",
            "pooling": "cls"
        }
    },
    {
        "name": "distilbert-zh", 
        "language": "chinese", 
        "args": {
            "model_type": "distilbert",
            "pooling": "mean"
        }
    },
    {
        "name": "hybrid-dBERTXLM", 
        "language": "chinese", 
        "args": {
            "model_type": "hybrid",
            "pooling": "mean",
            "hybrid_config": {
                "models": ("distilbert", "xlm-roberta"), 
                "weights": (0.4, 0.6)
            }
        }
    },
    {
        "name": "hybrid-SBXLM", 
        "language": "chinese", 
        "args": {
            "model_type": "hybrid",
            "pooling": "mean",
            "hybrid_config": {
                "models": ("sentence-bert", "xlm-roberta"), 
                "weights": (0.7, 0.3)
            }
        }
    }
]

# Embedding configurations for English
ENGLISH_EMBEDDING_CONFIGS = [
    {
        "name": "bert-base", 
        "language": "english", 
        "args": {
            "model_type": "bert-base",
            "pooling": "cls"  # Use CLS token pooling
        }
    },
    {
        "name": "roberta-base", 
        "language": "english", 
        "args": {
            "model_type": "roberta-base",
            "pooling": "cls"
        }
    },
    {
        "name": "distilbert-en", 
        "language": "english", 
        "args": {
            "model_type": "distilbert-en",
            "pooling": "mean"
        }
    },
    {
        "name": "sbert", 
        "language": "english", 
        "args": {
            "model_type": "sentence-bert",
            "pooling": "mean"  # Use mean pooling for Sentence BERT
        }
    },
    {
        "name": "hybrid-BERTRoBERTa", 
        "language": "english", 
        "args": {
            "model_type": "hybrid",
            "pooling": "mean",
            "hybrid_config": {
                "models": ("bert-base", "roberta-base"), 
                "weights": (0.5, 0.5)
            }
        }
    }
]

# =====================================================================
# AUTOENCODER CONFIGURATIONS
# =====================================================================

# Autoencoder processing parameters
AUTOENCODER_PARAMS = {
    "epochs": 100,         
    "batch_size": 64,     

}

# Method-specific autoencoder dimensions
METHOD_ENCODING_DIMS = {
    "bert-base": 128,      
    "roberta-base": 128,    
    "sbert": 128,           
    "distilbert-en": 96,    
    "cbert": 128,            
    "roberta-zh": 128,       
    "distilbert-zh": 96,   
    "hybrid-dBERTXLM": 160, 
    "hybrid-SBXLM": 160,     
    "hybrid-BERTRoBERTa": 160, 
    
    # Non-transformer methods
    "tfidf": 96,         
    "word2vec": 64           
}

# =====================================================================
# CLUSTERING CONFIGURATIONS
# =====================================================================

# Clustering parameters
CLUSTERING_PARAMS = {
    "k_range": (2, 15),  # Min and max number of clusters to try
    "n_init": 10,        # Number of initializations for k-means
    "random_state": 42   # Random seed for reproducibility
}