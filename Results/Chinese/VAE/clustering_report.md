# Clustering Analysis Report

Language: chinese
Autoencoder: vae

## Summary of Best Results

| Approach             | Category     |  K-means Best k |  K-means Score  | GMM Best k | GMM Score  |
|----------------------|--------------|-----------------|-----------------|------------|------------|
| roberta-zh           | BERT-family  |        2        |      0.3127     |     3      |   0.2934   |
| cbert                | BERT-family  |        2        |      0.2985     |     2      |   0.2985   |
| hybrid-dBERTXLM      | BERT-family  |        2        |      0.2722     |     2      |   0.2722   |
| distilbert-zh        | BERT-family  |        2        |      0.2686     |     2      |   0.2567   |
| sbert                | BERT-family  |        2        |      0.2237     |     2      |   0.1912   |
| hybrid-SBXLM         | XLM          |        3        |      0.0614     |     9      |   0.0510   |


## Analysis

The top-performing approaches were:

1. **roberta-zh** (BERT-family) with silhouette score 0.3127
2. **cbert** (BERT-family) with silhouette score 0.2985
3. **hybrid-dBERTXLM** (BERT-family) with silhouette score 0.2722

### Performance by Category

- **BERT-family**: Average silhouette score 0.2751 (K-means), 0.2624 (GMM)
- **XLM**: Average silhouette score 0.0614 (K-means), 0.0510 (GMM)
