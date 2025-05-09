"""
Utility functions for visualization and evaluation.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns

import os
import datetime

now = datetime.datetime.now()
# Format the datetime to be filename-safe
safe_now = now.strftime("%Y-%m-%d_%H-%M-%S")
output_dir = f"result"
os.makedirs(output_dir, exist_ok=True)

def cluster_and_visualize(latent_vectors, approach_name, language, autoencoder, k_range=(2, 14)):
    """Perform K-means and GMM clustering and visualize results."""
    # Apply PCA for visualization
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(latent_vectors)
    
    # Also try t-SNE for potentially better visualization
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latent_vectors)-1))
        tsne_vectors = tsne.fit_transform(latent_vectors)
    except Exception as e:
        print(f"t-SNE failed: {e}, using PCA only")
        tsne_vectors = None
    
    # Initialize results
    kmeans_results = {"best_k": 0, "best_score": -1, "labels": None}
    gmm_results = {"best_k": 0, "best_score": -1, "labels": None}
    
    # Try different k values
    for k in range(k_range[0], k_range[1] + 1):
        print(f"Trying k={k} for {approach_name}...")
        
        # K-means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(latent_vectors)
        
        # GMM
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
    
    # Create directory for output
    this_output_dir = os.path.join(output_dir, language, autoencoder, safe_now)
    os.makedirs(this_output_dir, exist_ok=True)
    
    # Plot the best clustering results with PCA
    plt.figure(figsize=(16, 7))
    
    # K-means visualization
    plt.subplot(1, 2, 1)
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=kmeans_results["labels"], cmap='tab10', alpha=0.7)
    plt.title(f'{approach_name} - K-means (k={kmeans_results["best_k"]}, score={kmeans_results["best_score"]:.4f})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    
    # GMM visualization
    plt.subplot(1, 2, 2)
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c=gmm_results["labels"], cmap='tab10', alpha=0.7)
    plt.title(f'{approach_name} - GMM (k={gmm_results["best_k"]}, score={gmm_results["best_score"]:.4f})')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(label='Cluster')
    
    plt.tight_layout()
    output_file = os.path.join(this_output_dir, f'clustering_pca_{approach_name}.png')
    plt.savefig(output_file)
    
    # If t-SNE was successful, create t-SNE visualization too
    if tsne_vectors is not None:
        plt.figure(figsize=(16, 7))
        
        # K-means visualization with t-SNE
        plt.subplot(1, 2, 1)
        plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=kmeans_results["labels"], cmap='tab10', alpha=0.7)
        plt.title(f'{approach_name} - K-means (k={kmeans_results["best_k"]}, score={kmeans_results["best_score"]:.4f})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(label='Cluster')
        
        # GMM visualization with t-SNE
        plt.subplot(1, 2, 2)
        plt.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], c=gmm_results["labels"], cmap='tab10', alpha=0.7)
        plt.title(f'{approach_name} - GMM (k={gmm_results["best_k"]}, score={gmm_results["best_score"]:.4f})')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(label='Cluster')
        
        plt.tight_layout()
        output_file = os.path.join(this_output_dir, f'clustering_tsne_{approach_name}.png')
        plt.savefig(output_file)
    
    print(f"Best K-means for {approach_name}: k={kmeans_results['best_k']}, score={kmeans_results['best_score']:.4f}")
    print(f"Best GMM for {approach_name}: k={gmm_results['best_k']}, score={gmm_results['best_score']:.4f}")
    
    return {"kmeans": kmeans_results, "gmm": gmm_results}

def compare_clustering_results(all_results, approaches, language, autoencoder):
    """Compare clustering results across different embedding approaches."""
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
    
    # Create output directory
    this_output_dir = os.path.join(output_dir, language, autoencoder, safe_now)
    os.makedirs(this_output_dir, exist_ok=True)
    
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
    plt.savefig(output_file)
    
    # Create a heatmap of results
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
    plt.savefig(output_file)
    
    # Print results in table format
    print("\nSummary of Best Results:")
    print("-" * 100)
    header = f"{'Approach':<20} | {'Category':<15} | {'K-means Best k':<15} | {'K-means Score':<15} | {'GMM Best k':<15} | {'GMM Score':<15}"
    print(header)
    print("-" * 100)
    
    # Sort approaches by K-means score for table
    approaches_sorted = sorted(approaches_found, 
                              key=lambda a: all_results[a]["kmeans"]["best_score"], 
                              reverse=True)
    
    for approach in approaches_sorted:
        category = categorize_approach(approach)
        kmeans_best_k = all_results[approach]["kmeans"]["best_k"]
        kmeans_score = all_results[approach]["kmeans"]["best_score"]
        gmm_best_k = all_results[approach]["gmm"]["best_k"]
        gmm_score = all_results[approach]["gmm"]["best_score"]
        
        row = f"{approach:<20} | {category:<15} | {kmeans_best_k:<15} | {kmeans_score:.4f}{' '*9} | {gmm_best_k:<15} | {gmm_score:.4f}"
        print(row)
    
    print("-" * 100)