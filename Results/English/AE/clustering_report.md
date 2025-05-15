# Clustering Analysis Report

Language: english
Autoencoder: ae

## Summary of Best Results

| Approach             | Category     |  K-means Best k |  K-means Score  | GMM Best k | GMM Score  |
|----------------------|--------------|-----------------|-----------------|------------|------------|
| sbert                | BERT-family  |        14       |      0.5426     |     13     |   0.4734   |
| distilbert-en        | BERT-family  |        10       |      0.4560     |     8      |   0.4217   |
| roberta-base         | BERT-family  |        10       |      0.4334     |     12     |   0.4238   |
| hybrid-BERTRoBERTa   | BERT-family  |        10       |      0.4273     |     9      |   0.4111   |
| bert-base            | BERT-family  |        12       |      0.3646     |     9      |   0.3336   |


## Analysis

The top-performing approaches were:

1. **sbert** (BERT-family) with silhouette score 0.5426
2. **distilbert-en** (BERT-family) with silhouette score 0.4560
3. **roberta-base** (BERT-family) with silhouette score 0.4334

### Performance by Category

- **BERT-family**: Average silhouette score 0.4448 (K-means), 0.4127 (GMM)
