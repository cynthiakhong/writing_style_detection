# Clustering Analysis Report

Language: chinese
Autoencoder: ae

## Summary of Best Results

| Approach             | Category     |  K-means Best k |  K-means Score  | GMM Best k | GMM Score  |
|----------------------|--------------|-----------------|-----------------|------------|------------|
| roberta-zh           | BERT-family  |        6        |      0.2564     |     6      |   0.2428   |
| cbert                | BERT-family  |        7        |      0.2215     |     8      |   0.1923   |
| hybrid-dBERTXLM      | BERT-family  |        13       |      0.1602     |     14     |   0.1367   |
| sbert                | BERT-family  |        11       |      0.1602     |     15     |   0.1378   |
| distilbert-zh        | BERT-family  |        13       |      0.1423     |     13     |   0.1423   |
| hybrid-SBXLM         | XLM          |        11       |      0.0749     |     11     |   0.0749   |


## Analysis

The top-performing approaches were:

1. **roberta-zh** (BERT-family) with silhouette score 0.2564
2. **cbert** (BERT-family) with silhouette score 0.2215
3. **hybrid-dBERTXLM** (BERT-family) with silhouette score 0.1602

### Performance by Category

- **BERT-family**: Average silhouette score 0.1881 (K-means), 0.1704 (GMM)
- **XLM**: Average silhouette score 0.0749 (K-means), 0.0749 (GMM)
