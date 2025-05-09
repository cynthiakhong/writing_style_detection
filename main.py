"""
Main script to run document analysis with options for Chinese or English processing.
"""
# Import configurations
import config

# Import processing modules
from process_data import load_data
from vectorize import vectorize, process_vectors 
from utils import cluster_and_visualize, compare_clustering_results

if __name__ == "__main__":

    assert config.LANGUAGE_CHOICE in ['cn', 'eng'], "Only cn and eng is acceptable"
    assert config.AUTOENCODER_CHOICE in ['vae', 'ae'], "Only ae and vae is acceptable"

    # Get language choice from config
    language_choice = config.LANGUAGE_CHOICE
    
    # Configure paths and settings based on language choice
    if language_choice == "cn":
        # Chinese documents
        print("Processing Chinese documents...")
        language = "chinese"
        
        # Get data path for Chinese documents from config
        data_path = config.CHINESE_DATA_PATH
        
        # Load the Chinese documents
        df = load_data(data_path, language=language)
        
        # Get embedding configurations for Chinese from config
        embedding_configs = config.CHINESE_EMBEDDING_CONFIGS

    elif language_choice == "eng":
        # English documents
        print("Processing English documents...")
        language = "english"
        
        # Get base data path and subdirectories from config
        data_path = config.BASE_DATA_PATH
        subdirs = config.ENGLISH_SUBDIRS
        
        # Load English data from multiple directories
        df = load_data(data_path, language=language, subdirs=subdirs)
        
        # Get embedding configurations for English from config
        embedding_configs = config.ENGLISH_EMBEDDING_CONFIGS
    
    # Create a list of all methods to use (including each transformer model separately)
    methods = []
    method_configs = {}
    
    # Add transformer models if specified in VECTORIZE_METHODS
    if 'transformers' in config.VECTORIZE_METHODS:
        # Add each transformer model as a separate method
        for embed_config in embedding_configs:
            model_name = embed_config["name"]
            methods.append(model_name)
            method_configs[model_name] = {
                'language': embed_config['language'],
                'model_type': embed_config.get('args', {}).get('model_type', model_name),
                **embed_config.get('args', {})
            }
            
            # For hybrid models, ensure hybrid_config is properly passed
            if 'hybrid_config' in embed_config.get('args', {}):
                method_configs[model_name]['hybrid_config'] = embed_config['args']['hybrid_config']
    
    # Add TF-IDF if specified
    if 'tfidf' in config.VECTORIZE_METHODS:
        methods.append('tfidf')
        method_configs['tfidf'] = config.TFIDF_CONFIG
    
    # Add Word2Vec if specified
    if 'word2vec' in config.VECTORIZE_METHODS:
        methods.append('word2vec')
        method_configs['word2vec'] = config.WORD2VEC_CONFIG
        
    # Use clustering parameters from config if available
    k_range = config.CLUSTERING_PARAMS.get('k_range', (2, 12))
    
    print(f"\nGenerating vectors using methods: {methods}")
    
    # Generate all vectors at once with the modified vectorize function
    vectors_dict = vectorize(
        df['text'],
        methods=methods,
        method_configs=method_configs
    )
    
    # Process each type of vector
    all_results = {}
    
    for method in methods:
        if method not in vectors_dict:
            print(f"Warning: No vectors found for method '{method}', skipping...")
            continue
        
        try:
            vectors = vectors_dict[method]
            
            print(f"\nProcessing {method} vectors with autoencoder...")
            # Get method-specific encoding dimension if available
            method_encoding_dim = config.METHOD_ENCODING_DIMS.get(
                method,  # Try exact match first
                config.AUTOENCODER_PARAMS.get("encoding_dim", 64)  # Default dimension
            )
            
            print(f"Using encoding dimension {method_encoding_dim} for method {method}")
            
            # Process vectors through autoencoder with appropriate parameters
            latent = process_vectors(
                name=method, 
                vectors=vectors,
                autoencoder=config.AUTOENCODER_CHOICE,  
                epochs=config.AUTOENCODER_PARAMS["epochs"],
                batch_size=config.AUTOENCODER_PARAMS["batch_size"],
                encoding_dim=method_encoding_dim
            )
            
            # Perform clustering and visualization with parameters from config
            print(f"\nAnalyzing {method} vectors...")
            results = cluster_and_visualize(
                latent_vectors=latent, 
                approach_name=method, 
                language=language, 
                autoencoder=config.AUTOENCODER_CHOICE,
                k_range=k_range
            )
            all_results[method] = results
            
        except Exception as e:
            print(f"Error processing {method} vectors: {e}")
    
    # Compare results across different vectorization approaches
    if len(all_results) > 1:
        print("\nComparing results across all approaches...")
        compare_clustering_results(
            all_results, 
            list(all_results.keys()), 
            language,
            config.AUTOENCODER_CHOICE
        )
    
    print("\nProcessing complete. See results in the output directories and generated visualizations.")