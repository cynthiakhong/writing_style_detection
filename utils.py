"""
Utility functions for visualization and evaluation.
"""
import os
import datetime
from typing import Dict, Tuple, List, Optional, Any, Union

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
import seaborn as sns

# Get current datetime for output directories
now = datetime.datetime.now()
# Format the datetime to be filename-safe
safe_now = now.strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"result"
os.makedirs(output_dir, exist_ok=True)

def get_output_dir(language, autoencoder, timestamp=None):
    """
    Create and return output directory path based on parameters.
    
    Args:
        language: Language being processed
        autoencoder: Autoencoder type being used
        timestamp: Optional timestamp (defaults to current time)
        
    Returns:
        Path to output directory
    """
    if timestamp is None:
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create specific directory path
    result_dir = os.path.join(output_dir, language, autoencoder, timestamp)
    os.makedirs(result_dir, exist_ok=True)
    
    return result_dir

def cluster_and_visualize(latent_vectors, approach_name, language, autoencoder, k_range=(2, 14)):
    """
    Perform K-means and GMM clustering and visualize results.
    
    Args:
        latent_vectors: Vectors to cluster
        approach_name: Name of the approach being used
        language: Language being processed
        autoencoder: Autoencoder type being used
        k_range: Range of cluster counts to try (min, max)
        
    Returns:
        Dictionary with clustering results
    """
    # Create output directory
    this_output_dir = os.path.join(output_dir, language, autoencoder, safe_now)
    os.makedirs(this_output_dir, exist_ok=True)
    
    if len(latent_vectors) < k_range[1]:
        print(f"Warning: Too few vectors ({len(latent_vectors)}) for k_range max of {k_range[1]}. Adjusting.")
        k_range = (min(k_range[0], len(latent_vectors) - 1), min(k_range[1], len(latent_vectors) - 1))
        if k_range[0] >= k_range[1]:
            k_range = (2, min(len(latent_vectors) - 1, 10))
            
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(latent_vectors)
    
    # Also try t-SNE for potentially better visualization
    try:
        tsne = TSNE(
            n_components=2, 
            random_state=42, 
            perplexity=min(30, max(5, len(latent_vectors)//10))
        )
        tsne_vectors = tsne.fit_transform(latent_vectors)
    except Exception as e:
        print(f"t-SNE failed: {e}, using PCA only")
        tsne_vectors = None
    
    # Initialize results dictionaries
    kmeans_results = {"best_k": 0, "best_score": -1, "labels": None}
    gmm_results = {"best_k": 0, "best_score": -1, "labels": None}
    
    # Try different k values
    for k in range(k_range[0], k_range[1] + 1):
        print(f"Trying k={k} for {approach_name}...")
        
        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(latent_vectors)
        
        # GMM clustering
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
        gmm_labels = gmm.fit_predict(latent_vectors)
        
        # Calculate silhouette scores
        if len(np.unique(kmeans_labels)) > 1:  # Ensure multiple clusters
            try:
                kmeans_silhouette = silhouette_score(latent_vectors, kmeans_labels)
                if kmeans_silhouette > kmeans_results["best_score"]:
                    kmeans_results["best_score"] = kmeans_silhouette
                    kmeans_results["best_k"] = k
                    kmeans_results["labels"] = kmeans_labels
            except Exception as e:
                print(f"Error calculating K-means silhouette for k={k}: {e}")
        
        if len(np.unique(gmm_labels)) > 1:  # Ensure multiple clusters
            try:
                gmm_silhouette = silhouette_score(latent_vectors, gmm_labels)
                if gmm_silhouette > gmm_results["best_score"]:
                    gmm_results["best_score"] = gmm_silhouette
                    gmm_results["best_k"] = k
                    gmm_results["labels"] = gmm_labels
            except Exception as e:
                print(f"Error calculating GMM silhouette for k={k}: {e}")
    
    # Plot the best clustering results with PCA
    plt.figure(figsize=(16, 7))
    
    # K-means visualization
    plt.subplot(1, 2, 1)
    plt.scatter(
        reduced_vectors[:, 0], reduced_vectors[:, 1], 
        c=kmeans_results["labels"], 
        cmap='tab10', 
        alpha=0.7
    )
    plt.title(f'{approach_name} - K-means (k={kmeans_results["best_k"]}, score={kmeans_results["best_score"]:.4f})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.colorbar(label='Cluster')
    
    # GMM visualization
    plt.subplot(1, 2, 2)
    plt.scatter(
        reduced_vectors[:, 0], reduced_vectors[:, 1], 
        c=gmm_results["labels"], 
        cmap='tab10', 
        alpha=0.7
    )
    plt.title(f'{approach_name} - GMM (k={gmm_results["best_k"]}, score={gmm_results["best_score"]:.4f})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.colorbar(label='Cluster')
    
    plt.tight_layout()
    output_file = os.path.join(this_output_dir, f'clustering_pca_{approach_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # If t-SNE was successful, create t-SNE visualization too
    if tsne_vectors is not None:
        plt.figure(figsize=(16, 7))
        
        # K-means visualization with t-SNE
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(
            tsne_vectors[:, 0], tsne_vectors[:, 1], 
            c=kmeans_results["labels"], 
            cmap='tab10', 
            alpha=0.7
        )
        plt.title(f'{approach_name} - K-means (k={kmeans_results["best_k"]}, score={kmeans_results["best_score"]:.4f})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend if not too many clusters
        if kmeans_results["best_k"] <= 10:
            legend1 = plt.legend(*scatter.legend_elements(),
                                title="Clusters", loc="upper right")
            plt.gca().add_artist(legend1)
        else:
            plt.colorbar(label='Cluster')
        
        # GMM visualization with t-SNE
        plt.subplot(1, 2, 2)
        scatter = plt.scatter(
            tsne_vectors[:, 0], tsne_vectors[:, 1], 
            c=gmm_results["labels"], 
            cmap='tab10', 
            alpha=0.7
        )
        plt.title(f'{approach_name} - GMM (k={gmm_results["best_k"]}, score={gmm_results["best_score"]:.4f})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.grid(True, linestyle='--', alpha=0.3)
        
        # Add legend if not too many clusters
        if gmm_results["best_k"] <= 10:
            legend1 = plt.legend(*scatter.legend_elements(),
                                title="Clusters", loc="upper right")
            plt.gca().add_artist(legend1)
        else:
            plt.colorbar(label='Cluster')
        
        plt.tight_layout()
        output_file = os.path.join(this_output_dir, f'clustering_tsne_{approach_name}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"Best K-means for {approach_name}: k={kmeans_results['best_k']}, score={kmeans_results['best_score']:.4f}")
    print(f"Best GMM for {approach_name}: k={gmm_results['best_k']}, score={gmm_results['best_score']:.4f}")
    
    return {"kmeans": kmeans_results, "gmm": gmm_results}

def compare_clustering_results(all_results, approaches, language, autoencoder):
    """
    Compare clustering results across different embedding approaches.
    
    Args:
        all_results: Dictionary of results for each approach
        approaches: List of approach names
        language: Language being processed
        autoencoder: Autoencoder type being used
    """
    # Create output directory
    this_output_dir = os.path.join(output_dir, language, autoencoder, safe_now)
    os.makedirs(this_output_dir, exist_ok=True)
    
    # Filter out approaches with no results
    approaches_found = [a for a in approaches if a in all_results]
    
    if not approaches_found:
        print("No valid approaches to compare.")
        return
    
    # Group the approaches by their general category for better visualization
    def categorize_approach(approach):
        if any(t in approach.lower() for t in ['bert', 'roberta']):
            return 'BERT-family'
        elif 'sentence' in approach.lower() or 'sbert' in approach.lower():
            return 'Sentence-BERT'
        elif 'distil' in approach.lower():
            return 'DistilBERT'
        elif 'xlm' in approach.lower():
            return 'XLM'
        elif 'word2vec' in approach.lower():
            return 'Word2Vec'
        elif 'tfidf' in approach.lower():
            return 'TF-IDF'
        else:
            return 'Other'
    
    # Sort approaches by category and name
    approaches_found.sort(key=lambda x: (categorize_approach(x), x))
    
    # Extract best scores
    kmeans_best_scores = [all_results[a]["kmeans"]["best_score"] for a in approaches_found]
    gmm_best_scores = [all_results[a]["gmm"]["best_score"] for a in approaches_found]
    
    # Create bar chart
    plt.figure(figsize=(14, 8))
    x = np.arange(len(approaches_found))
    width = 0.35
    
    # Color by category
    colors = {
        'BERT-family': '#1f77b4',
        'Sentence-BERT': '#ff7f0e',
        'DistilBERT': '#2ca02c',
        'XLM': '#d62728',
        'Word2Vec': '#9467bd',
        'TF-IDF': '#8c564b',
        'Other': '#e377c2'
    }
    
    # Get colors for each approach
    kmeans_colors = [colors[categorize_approach(a)] for a in approaches_found]
    gmm_colors = [sns.desaturate(colors[categorize_approach(a)], 0.7) for a in approaches_found]
    
    bars1 = plt.bar(x - width/2, kmeans_best_scores, width, label='K-means', color=kmeans_colors)
    bars2 = plt.bar(x + width/2, gmm_best_scores, width, label='GMM', color=gmm_colors)
    
    plt.xlabel('Embedding Approach', fontsize=12)
    plt.ylabel('Best Silhouette Score', fontsize=12)
    plt.title('Best Clustering Performance by Approach', fontsize=14)
    plt.xticks(x, approaches_found, rotation=45, ha='right', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add value labels on the bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    # Add labels with best k values
    for i, v in enumerate(kmeans_best_scores):
        plt.text(i - width/2, v + 0.03, f'k={all_results[approaches_found[i]]["kmeans"]["best_k"]}', 
                 ha='center', va='bottom', fontsize=8)
        
    for i, v in enumerate(gmm_best_scores):
        plt.text(i + width/2, v + 0.03, f'k={all_results[approaches_found[i]]["gmm"]["best_k"]}', 
                 ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    output_file = os.path.join(this_output_dir, f'best_score_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a method ranking visualization
    plt.figure(figsize=(12, 8))
    scores_dict = {
        f"{a} K-means (k={all_results[a]['kmeans']['best_k']})": all_results[a]["kmeans"]["best_score"]
        for a in approaches_found
    }
    scores_dict.update({
        f"{a} GMM (k={all_results[a]['gmm']['best_k']})": all_results[a]["gmm"]["best_score"]
        for a in approaches_found
    })
    
    # Sort by score values
    sorted_items = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
    methods = [item[0] for item in sorted_items]
    scores = [item[1] for item in sorted_items]
    
    # Create horizontal bar chart for ranking
    plt.barh(methods, scores, color=sns.color_palette("viridis", len(methods)))
    plt.xlabel('Silhouette Score')
    plt.title('Ranking of All Methods by Silhouette Score')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(this_output_dir, f'method_ranking.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Sort approaches by K-means score for table
    approaches_sorted = sorted(approaches_found, 
                              key=lambda a: all_results[a]["kmeans"]["best_score"], 
                              reverse=True)
    
    # Define fixed column widths for consistent formatting
    col_widths = {
        'approach': 20,       # Width for approach column
        'category': 12,       # Width for category column
        'kmeans_k': 15,       # Width for K-means Best k column
        'kmeans_score': 15,   # Width for K-means Score column
        'gmm_k': 10,          # Width for GMM Best k column
        'gmm_score': 10       # Width for GMM Score column
    }
    
    # Create detailed report in markdown format
    report_file = os.path.join(this_output_dir, 'clustering_report.md')
    
    with open(report_file, 'w') as f:
        f.write(f"# Clustering Analysis Report\n\n")
        f.write(f"Language: {language}\n")
        f.write(f"Autoencoder: {autoencoder}\n\n")
        
        f.write("## Summary of Best Results\n\n")
        
        # Create perfectly aligned table header
        header_line = f"| {'Approach'.ljust(col_widths['approach'])} | {'Category'.ljust(col_widths['category'])} | {'K-means Best k'.center(col_widths['kmeans_k'])} | {'K-means Score'.center(col_widths['kmeans_score'])} | {'GMM Best k'.center(col_widths['gmm_k'])} | {'GMM Score'.center(col_widths['gmm_score'])} |"
        f.write(header_line + "\n")
        
        # Create separator line with exact same width as header
        separator_line = f"|{'-' * (col_widths['approach'] + 2)}|{'-' * (col_widths['category'] + 2)}|{'-' * (col_widths['kmeans_k'] + 2)}|{'-' * (col_widths['kmeans_score'] + 2)}|{'-' * (col_widths['gmm_k'] + 2)}|{'-' * (col_widths['gmm_score'] + 2)}|"
        f.write(separator_line + "\n")
        
        # Write rows with proper spacing
        for approach in approaches_sorted:
            category = categorize_approach(approach)
            kmeans_best_k = all_results[approach]["kmeans"]["best_k"]
            kmeans_score = all_results[approach]["kmeans"]["best_score"]
            gmm_best_k = all_results[approach]["gmm"]["best_k"]
            gmm_score = all_results[approach]["gmm"]["best_score"]
            
            row = f"| {approach.ljust(col_widths['approach'])} | {category.ljust(col_widths['category'])} | {str(kmeans_best_k).center(col_widths['kmeans_k'])} | {f'{kmeans_score:.4f}'.center(col_widths['kmeans_score'])} | {str(gmm_best_k).center(col_widths['gmm_k'])} | {f'{gmm_score:.4f}'.center(col_widths['gmm_score'])} |"
            f.write(row + "\n")
        
        f.write("\n\n## Analysis\n\n")
        f.write("The top-performing approaches were:\n\n")
        
        # List top 3 approaches
        top_n = min(3, len(approaches_sorted))
        for i, approach in enumerate(approaches_sorted[:top_n]):
            f.write(f"{i+1}. **{approach}** ({categorize_approach(approach)}) with silhouette score {all_results[approach]['kmeans']['best_score']:.4f}\n")
        
        # Add observations about categories
        f.write("\n### Performance by Category\n\n")
        
        # Group by category and calculate average scores
        category_scores = {}
        for approach in approaches_found:
            category = categorize_approach(approach)
            if category not in category_scores:
                category_scores[category] = {"kmeans": [], "gmm": []}
            
            category_scores[category]["kmeans"].append(all_results[approach]["kmeans"]["best_score"])
            category_scores[category]["gmm"].append(all_results[approach]["gmm"]["best_score"])
        
        for category, scores in sorted(category_scores.items(), 
                                     key=lambda x: np.mean(x[1]["kmeans"]), 
                                     reverse=True):
            avg_kmeans = np.mean(scores["kmeans"])
            avg_gmm = np.mean(scores["gmm"])
            f.write(f"- **{category}**: Average silhouette score {avg_kmeans:.4f} (K-means), {avg_gmm:.4f} (GMM)\n")
    
    # Print results in table format with consistent spacing
    print("\nSummary of Best Results:")
    print("-" * 100)
    
    # Format the header with proper spacing for console output
    console_header = f"| {'Approach'.ljust(col_widths['approach'])} | {'Category'.ljust(col_widths['category'])} | {'K-means Best k'.ljust(col_widths['kmeans_k'])} | {'K-means Score'.ljust(col_widths['kmeans_score'])} | {'GMM Best k'.ljust(col_widths['gmm_k'])} | {'GMM Score'.ljust(col_widths['gmm_score'])} |"
    print(console_header)
    print("-" * 100)
    
    for approach in approaches_sorted:
        category = categorize_approach(approach)
        kmeans_best_k = all_results[approach]["kmeans"]["best_k"]
        kmeans_score = all_results[approach]["kmeans"]["best_score"]
        gmm_best_k = all_results[approach]["gmm"]["best_k"]
        gmm_score = all_results[approach]["gmm"]["best_score"]
        
        # Format the row with proper spacing for console output
        console_row = f"| {approach.ljust(col_widths['approach'])} | {category.ljust(col_widths['category'])} | {str(kmeans_best_k).ljust(col_widths['kmeans_k'])} | {f'{kmeans_score:.4f}'.ljust(col_widths['kmeans_score'])} | {str(gmm_best_k).ljust(col_widths['gmm_k'])} | {f'{gmm_score:.4f}'.ljust(col_widths['gmm_score'])} |"
        print(console_row)
    
    print("-" * 100)
    
    return this_output_dir