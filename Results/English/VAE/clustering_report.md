# Clustering Analysis Report

Language: english
Autoencoder: vae

## Summary of Best Results

| Approach             | Category     |  K-means Best k |  K-means Score  | GMM Best k | GMM Score  |
|----------------------|--------------|-----------------|-----------------|------------|------------|
| sbert                | BERT-family  |        14       |      0.7099     |     11     |   0.6699   |
| distilbert-en        | BERT-family  |        8        |      0.5968     |     8      |   0.5639   |
| hybrid-BERTRoBERTa   | BERT-family  |        11       |      0.5386     |     12     |   0.5424   |
| roberta-base         | BERT-family  |        9        |      0.5241     |     8      |   0.5208   |
| bert-base            | BERT-family  |        7        |      0.5214     |     7      |   0.5214   |
| word2vec             | Word2Vec     |        10       |      0.3217     |     11     |   0.3030   |
| tfidf                | TF-IDF       |        13       |      0.1747     |     15     |   0.1477   |


## Analysis

The top-performing approaches were:

1. **sbert** (BERT-family) with silhouette score 0.7099
2. **distilbert-en** (BERT-family) with silhouette score 0.5968
3. **hybrid-BERTRoBERTa** (BERT-family) with silhouette score 0.5386

### Performance by Category

- **BERT-family**: Average silhouette score 0.5781 (K-means), 0.5637 (GMM)
- **Word2Vec**: Average silhouette score 0.3217 (K-means), 0.3030 (GMM)
- **TF-IDF**: Average silhouette score 0.1747 (K-means), 0.1477 (GMM)
