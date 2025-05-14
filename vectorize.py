"""
Models for embedding generation, vectorization, and dimensionality reduction.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from autoencoders import Autoencoder, VAE
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api

# Initialize device once
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(text_input, language="chinese", model_type="chinese-bert", pooling="cls", batch_size=16, hybrid_config=None):
    """
    Unified embedding function supporting both Chinese and English models,
    including support for hybrid models.
    
    Args:
        text_input: Input text (str, list, or pandas.Series)
        language: "chinese" or "english"
        model_type: Model type based on language
        pooling: Pooling strategy - "cls", "mean", or "max"
        batch_size: Batch size for processing
        hybrid_config: Optional configuration for hybrid models with format:
                      {"models": (model1, model2), "weights": (weight1, weight2)}
        
    Returns:
        numpy array: Embedding vectors
    """
    # Standardize input
    if isinstance(text_input, pd.Series):
        text_list = text_input.fillna("").astype(str).tolist()
    elif isinstance(text_input, str):
        text_list = [text_input]
    else:
        text_list = [str(t) if t is not None else "" for t in text_input]
    
    # Define model configurations
    chinese_models = {
        "chinese-bert": {"name": "hfl/chinese-bert-wwm-ext"},
        "bert-base": {"name": "hfl/chinese-bert-wwm-ext"},  # Alias
        "cbert": {"name": "hfl/chinese-bert-wwm-ext"},      # Alias
        "sentence-bert": {"name": "sentence-transformers/distiluse-base-multilingual-cased-v2"},
        "sbert": {"name": "sentence-transformers/distiluse-base-multilingual-cased-v2"},  # Alias
        "roberta": {"name": "hfl/chinese-roberta-wwm-ext"},
        "robertaexe": {"name": "hfl/chinese-roberta-wwm-ext"},  # Alias
        "xlm-roberta": {"name": "xlm-roberta-base"},
        "distilbert": {"name": "distilbert-base-multilingual-cased"}
    }
    
    english_models = {
        "bert-base": {"name": "bert-base-uncased"},
        "roberta-base": {"name": "roberta-base"},
        "distilbert": {"name": "distilbert-base-uncased"},
        "distilbert-en": {"name": "distilbert-base-uncased"},  # Alias
        "sentence-bert": {"name": "sentence-transformers/all-mpnet-base-v2"},
        "sbert": {"name": "sentence-transformers/all-mpnet-base-v2"},  # Alias
        "mpnet": {"name": "sentence-transformers/all-mpnet-base-v2"},  # Alias
    }
    
    # Select the appropriate model config based on language
    model_config = chinese_models if language == "chinese" else english_models

    # Special case for hybrid models
    if hybrid_config:
        return _get_hybrid_embeddings(
            text_list, 
            language, 
            model_config, 
            hybrid_config,
            pooling, 
            batch_size
        )
    
    # Check if model type is valid
    if model_type.lower() not in model_config:
        closest_match = _get_closest_match(model_type.lower(), model_config.keys())
        if closest_match:
            print(f"Warning: Model type '{model_type}' not found. Using closest match '{closest_match}'")
            model_type = closest_match
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Available models: {list(model_config.keys())}")
    
    model_name = model_config[model_type.lower()]["name"]
    
    # Load model (without caching)
    print(f"Loading {model_name} model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)
    
    print(f"Processing with {model_name} on {device}")
    
    # Process in batches
    all_embeddings = []
    for i in tqdm(range(0, len(text_list), batch_size), desc=f"Encoding with {model_type}"):
        batch_texts = text_list[i:i+batch_size]
        
        # Tokenize
        encoded_input = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True,
            max_length=512, 
            return_tensors='pt'
        )
        
        # Move to device
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = model(**encoded_input)
        
        # Apply pooling strategy
        last_hidden_state = outputs.last_hidden_state
        attention_mask = encoded_input['attention_mask'].unsqueeze(-1)
        
        if pooling == "cls":
            batch_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
        elif pooling == "mean":
            sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
            sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
            batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
        elif pooling == "max":
            masked = last_hidden_state * attention_mask
            masked[attention_mask == 0] = -1e9
            batch_embeddings = torch.max(masked, dim=1)[0].cpu().numpy()
        
        all_embeddings.append(batch_embeddings)
    
    # Combine results
    if all_embeddings:
        final_embeddings = np.vstack(all_embeddings)
    else:
        final_embeddings = np.array([])
    
    # Return single embedding for single input
    if isinstance(text_input, str):
        return final_embeddings[0]
    
    return final_embeddings

def _get_hybrid_embeddings(text_list, language, model_config, hybrid_config, pooling="cls", batch_size=16):
    """
    Helper function to create hybrid embeddings by combining multiple models.
    """
    models = hybrid_config['models']
    weights = hybrid_config.get('weights', [1.0/len(models)] * len(models))
    
    if len(models) != len(weights):
        raise ValueError("Number of models must match number of weights")
    
    # Normalize weights
    weights = np.array(weights) / sum(weights)
    
    print(f"Creating hybrid embeddings with models: {models} and weights: {weights}")
    
    # Get embeddings for each model
    all_model_embeddings = []
    
    for i, model_type in enumerate(models):
        if model_type not in model_config:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_name = model_config[model_type]["name"]
        weight = weights[i]
        
        print(f"Processing model {i+1}/{len(models)}: {model_type} (weight: {weight:.2f})")
        
        # Load model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        model.to(device)
        
        # Process in batches
        model_embeddings = []
        for j in tqdm(range(0, len(text_list), batch_size), desc=f"Encoding with {model_type}"):
            batch_texts = text_list[j:j+batch_size]
            
            # Tokenize
            encoded_input = tokenizer(
                batch_texts, padding=True, truncation=True,
                max_length=512, return_tensors='pt'
            )
            
            # Move to device
            encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = model(**encoded_input)
            
            # Apply pooling strategy
            last_hidden_state = outputs.last_hidden_state
            attention_mask = encoded_input['attention_mask'].unsqueeze(-1)
            
            if pooling == "cls":
                batch_embeddings = last_hidden_state[:, 0, :].cpu().numpy()
            elif pooling == "mean":
                sum_embeddings = torch.sum(last_hidden_state * attention_mask, 1)
                sum_mask = torch.clamp(attention_mask.sum(1), min=1e-9)
                batch_embeddings = (sum_embeddings / sum_mask).cpu().numpy()
            elif pooling == "max":
                masked = last_hidden_state * attention_mask
                masked[attention_mask == 0] = -1e9
                batch_embeddings = torch.max(masked, dim=1)[0].cpu().numpy()
            
            model_embeddings.append(batch_embeddings)
        
        # Combine results for this model
        if model_embeddings:
            model_embeddings = np.vstack(model_embeddings)
            # Normalize to unit length
            norms = np.linalg.norm(model_embeddings, axis=1, keepdims=True)
            model_embeddings = model_embeddings / np.maximum(norms, 1e-6)
            all_model_embeddings.append((model_embeddings, weight))
    
    # Combine all models with weighted average
    # First, check if all embeddings have the same shape
    embedding_shapes = [emb.shape for emb, _ in all_model_embeddings]
    if len(set(shape[0] for shape in embedding_shapes)) > 1:
        raise ValueError("All models must produce the same number of embeddings")
    
    # We need to handle different embedding dimensions by projecting to a common space
    # For simplicity, we'll concatenate all embeddings with their respective weights
    final_embeddings = np.zeros((embedding_shapes[0][0], 
                                np.sum([shape[1] for shape in embedding_shapes])))
    
    start_idx = 0
    for embeddings, weight in all_model_embeddings:
        end_idx = start_idx + embeddings.shape[1]
        final_embeddings[:, start_idx:end_idx] = embeddings * weight
        start_idx = end_idx
    
    return final_embeddings

def _get_closest_match(query, options):
    """Find the closest matching model name."""
    # Simple fuzzy matching - could be improved with more sophisticated algorithms
    for option in options:
        if query in option or option in query:
            return option
    return None

def get_tfidf_vectors(text_input, max_features=10000, tokenizer=None):
    """
    Create TF-IDF vectors from text input.
    
    Args:
        text_input: Input text (str, list, or pandas.Series)
        max_features: Maximum number of features for TF-IDF
        tokenizer: Optional pre-trained tokenizer to use for text preprocessing
        
    Returns:
        numpy.ndarray: TF-IDF vectors
    """
    print(f"Creating TF-IDF vectors with max_features={max_features}...")
    
    # Standardize input
    if isinstance(text_input, pd.Series):
        text_list = text_input.fillna("").astype(str).tolist()
    elif isinstance(text_input, str):
        text_list = [text_input]
    else:
        text_list = [str(t) if t is not None else "" for t in text_input]
    
    # Create vectorizer
    tfidf_vectorizer = TfidfVectorizer(max_features=max_features)
    
    # If input is just one string, we need to return the first row
    is_single_input = isinstance(text_input, str)
    
    # Fit and transform
    tfidf_vectors = tfidf_vectorizer.fit_transform(text_list)
    
    # Convert to dense array for consistency with other methods
    tfidf_vectors_dense = tfidf_vectors.toarray()
    
    # Return single vector for single input
    if is_single_input:
        return tfidf_vectors_dense[0]
    
    return tfidf_vectors_dense

def get_word2vec_vectors(text_input, model_name='word2vec-google-news-300', custom_model=None):
    """
    Create Word2Vec embeddings by averaging word vectors.
    
    Args:
        text_input: Input text (str, list, or pandas.Series)
        model_name: Pre-trained model to load from gensim
        custom_model: Optional custom Word2Vec model
        
    Returns:
        numpy.ndarray: Word2Vec embedding vectors
    """
    print(f"Creating Word2Vec vectors using {model_name} model...")
    
    # Standardize input
    if isinstance(text_input, pd.Series):
        text_list = text_input.fillna("").astype(str).tolist()
    elif isinstance(text_input, str):
        text_list = [text_input]
    else:
        text_list = [str(t) if t is not None else "" for t in text_input]
    
    # Load or use model
    if custom_model:
        model = custom_model
    else:
        model = api.load(model_name)
    
    vector_size = model.vector_size
    print(f"Using Word2Vec model with vector_size={vector_size}")
    
    # Document embedding function
    def get_document_embedding(text, model):
        # Tokenize text
        words = text.lower().split()
        # Filter words in vocabulary and get their vectors
        word_vectors = [model[word] for word in words if word in model]
        if len(word_vectors) == 0:
            return np.zeros(model.vector_size)
        # Average the word vectors
        return np.mean(word_vectors, axis=0)
    
    # Process documents with progress bar
    all_vectors = []
    for text in tqdm(text_list, desc="Creating Word2Vec embeddings"):
        all_vectors.append(get_document_embedding(text, model))
    
    # Stack all vectors
    vectors_array = np.vstack(all_vectors)
    
    # Return single vector for single input
    if isinstance(text_input, str):
        return vectors_array[0]
    
    return vectors_array

def vectorize(text_input, methods=None, method_configs=None, transformer_models=None):
    """
    Unified vectorization function that supports multiple methods at once,
    with explicit support for different transformer models.
    
    Args:
        text_input: Input text (str, list, or pandas.Series)
        methods: List of vectorization methods - ["tfidf", "word2vec", etc.]
                 For transformers, specify the model name directly (e.g., "bert-base", "sbert")
                 If None, defaults to ["transformer"]
        method_configs: Dictionary of configurations for each method
                       Format: {"method_name": {**kwargs}}
        transformer_models: List of transformer model configs from config (deprecated, use methods instead)
    
    Returns:
        dict: Dictionary of vector representations keyed by method name
    """
    if methods is None:
        methods = ["transformer"]
    
    if method_configs is None:
        method_configs = {}
    
    # Initialize results dictionary
    results = {}
    
    # Process each method
    for method in methods:
        print(f"\nGenerating vectors using {method} method...")
        
        # Get method-specific configs (or empty dict if not provided)
        config = method_configs.get(method, {})
        
        # Detect if method is a transformer type based on name patterns
        is_transformer = any(t_type in method.lower() for t_type in 
                             ["bert", "roberta", "xlm", "distil", "sentence", "transformer"])
        
        # Route to appropriate method
        if is_transformer:
            # For transformer models, pass the method name as model_type
            transformer_config = config.copy()
            if 'model_type' not in transformer_config:
                transformer_config['model_type'] = method
                
            results[method] = get_embeddings(text_input, **transformer_config)
    
        elif method == "tfidf":
            # TF-IDF vectorization
            max_features = config.get("max_features", 10000)
            tokenizer = config.get("tokenizer", None)
            results[method] = get_tfidf_vectors(text_input, max_features, tokenizer)
        
        elif method == "word2vec":
            # Word2Vec vectorization
            model_name = config.get("model_name", "word2vec-google-news-300")
            custom_model = config.get("custom_model", None)
            results[method] = get_word2vec_vectors(text_input, model_name, custom_model)
        
        else:
            print(f"Warning: Unsupported vectorization method: {method}, skipping...")
            continue
            
        # Print shape information
        if method in results and results[method] is not None:
            print(f"  {method} vectors shape: {results[method].shape}")
    
    return results

def process_vectors(name, vectors, autoencoder, epochs=100, batch_size=32, encoding_dim=64):
    """Process embeddings through an autoencoder and return latent representations"""
    print(f"Processing {name} embeddings...")
    
    # Convert to numpy array if not already
    if not isinstance(vectors, np.ndarray):
        try:
            vectors_array = np.array(vectors)
        except Exception as e:
            print(f"Error converting to numpy array: {e}")
            if isinstance(vectors, list) and len(vectors) > 0:
                if hasattr(vectors[0], 'numpy'):
                    # Try converting torch tensors to numpy
                    vectors_array = np.vstack([v.numpy() for v in vectors])
                else:
                    # Try stacking arrays
                    vectors_array = np.vstack([np.array(v) for v in vectors])
            else:
                raise ValueError("Could not convert input to numpy array")
    else:
        vectors_array = vectors
    
    # Ensure 2D shape
    if len(vectors_array.shape) != 2:
        try:
            vectors_array = vectors_array.reshape(vectors_array.shape[0], -1)
        except Exception as e:
            print(f"Reshape error: {e}")
            raise ValueError(f"Could not reshape array with shape {vectors_array.shape}")
    
    # Clean data
    vectors_array = np.nan_to_num(vectors_array)
    
    # Print data stats
    print(f"Data shape: {vectors_array.shape}")
    print(f"Original data range: [{vectors_array.min():.4f}, {vectors_array.max():.4f}]")
    
    # Normalize data
    scaler = StandardScaler()
    vectors_normalized = scaler.fit_transform(vectors_array)
    print(f"After normalization range: [{vectors_normalized.min():.4f}, {vectors_normalized.max():.4f}]")
    
    # Create tensor dataset
    tensor_x = torch.FloatTensor(vectors_normalized)
    dataset = TensorDataset(tensor_x)
    
    # Adjust batch size if needed
    actual_batch_size = min(batch_size, len(dataset))
    print(f"Using batch size: {actual_batch_size}")
    
    # Create data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=actual_batch_size,
        shuffle=True,
        drop_last=False  # Changed to False to include all data
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Automatically adjust encoding_dim based on input dimensionality
    # For high-dim vectors like BERT (768), use larger latent space
    input_dim = vectors_normalized.shape[1]
    if encoding_dim is None:
        if input_dim >= 512:
            encoding_dim = min(256, input_dim // 3)
        elif input_dim >= 300:
            encoding_dim = min(128, input_dim // 2)
        else:
            encoding_dim = min(64, input_dim // 2)
            
    print(f"Using encoding dimension: {encoding_dim} for input dimension: {input_dim}")
    
    # Determine model architecture based on input name
    if 'bert' in name.lower() or 'roberta' in name.lower():
        # For Transformer models, use larger hidden dimensions
        hidden_dim = min(512, input_dim)
    else:
        # For other models like TF-IDF, Word2Vec
        hidden_dim = min(256, input_dim)

    # Create the model
    if autoencoder == 'vae':
        model = VAE(
            input_dim=input_dim, 
            encoding_dim=encoding_dim,
            hidden_dim=min(256, input_dim),  # Reduce hidden_dim
            dropout_rate=0.3 if input_dim > 300 else 0.1  # Higher dropout for high-dim inputs
        )
    else:
        model = Autoencoder(
            input_dim=input_dim, 
            encoding_dim=encoding_dim,
            dropout_rate=0.2 if input_dim > 300 else 0.1
        )

    model.to(device)
    
    # Set loss and optimizer
    criterion = nn.MSELoss()
    
    # Adjust learning rate based on data characteristics and model type
    if 'bert' in name.lower() or 'transformer' in name.lower():
        lr = 0.0001  # Lower learning rate for transformer embeddings
    else:
        lr = 0.001   # Higher learning rate for other embeddings
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    # Training loop
    print(f"Starting training for {name} with {epochs} epochs...")
    best_loss = float('inf')
    best_model_state = None
    patience = 15  # Early stopping patience
    no_improve_count = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch in dataloader:
            inputs = batch[0].to(device)
            
            if autoencoder == 'vae':
                # For VAE
                outputs, mu, logvar = model(inputs)
                # Use VAE's loss function which includes KL divergence
                loss, recon_loss, kl_loss = VAE.vae_loss(outputs, inputs, mu, logvar, kl_weight=0.5)
                if epoch == 0 or (epoch + 1) % 10 == 0:
                    # Print detailed loss components occasionally
                    print(f"  Epoch {epoch+1} - Total: {loss:.4f}, Recon: {recon_loss:.4f}, KL: {kl_loss:.4f}")
            else:
                # For regular Autoencoder
                outputs, _ = model(inputs)
                loss = criterion(outputs, inputs)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss/len(dataloader)
        scheduler.step(epoch_loss)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        # Early stopping
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch+1} due to no improvement")
            break
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Get encoded representations
    model.eval()
    with torch.no_grad():
        encoded_data = []
        for batch in dataloader:
            inputs = batch[0].to(device)
            
            if autoencoder == 'vae':
                encoded = model.get_latent(inputs)
            else:
                encoded = model.encode(inputs)
            encoded_data.append(encoded.cpu().numpy())
        
        # Concatenate all batches
        if len(encoded_data) > 0:
            latent_vectors = np.vstack(encoded_data)
        else:
            latent_vectors = np.array([])
    
    print(f"Encoding complete. Latent shape: {latent_vectors.shape}")
    print(f"Final loss: {best_loss:.6f}")
    
    # Visualize reconstruction quality on a small sample
    if len(latent_vectors) > 0:
        # Take a small sample for visualization
        sample_size = min(5, len(dataset))
        sample_indices = np.random.choice(len(dataset), sample_size, replace=False)
        sample_inputs = torch.stack([dataset[i][0] for i in sample_indices]).to(device)
        
        # Get reconstructions
        if autoencoder == 'vae':
            sample_outputs, _, _ = model(sample_inputs)
        else:
            sample_outputs, _ = model(sample_inputs)
            
        # Convert to numpy for plotting
        sample_inputs_np = sample_inputs.cpu().numpy()
        sample_outputs_np = sample_outputs.detach().cpu().numpy()
        
        # Calculate reconstruction error
        mse = np.mean((sample_inputs_np - sample_outputs_np) ** 2, axis=1)
        print(f"Sample reconstruction MSE: {mse.mean():.6f}")
    
    return latent_vectors