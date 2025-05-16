# Clustering Analysis Report

Language: english
Autoencoder: ae

## Summary of Best Results

| Approach             | Category     |  K-means Best k |  K-means Score  | GMM Best k | GMM Score  |
|----------------------|--------------|-----------------|-----------------|------------|------------|
| sbert                | BERT-family  |        12       |      0.5058     |     10     |   0.4733   |
| distilbert-en        | BERT-family  |        11       |      0.4785     |     9      |   0.4586   |
| hybrid-BERTRoBERTa   | BERT-family  |        15       |      0.4553     |     8      |   0.3873   |
| roberta-base         | BERT-family  |        10       |      0.4205     |     9      |   0.4071   |
| word2vec             | Word2Vec     |        8        |      0.3577     |     7      |   0.3510   |
| bert-base            | BERT-family  |        12       |      0.3419     |     15     |   0.3373   |
| tfidf                | TF-IDF       |        14       |      0.2133     |     14     |   0.1764   |


## Analysis

The top-performing approaches were:

1. **sbert** (BERT-family) with silhouette score 0.5058
2. **distilbert-en** (BERT-family) with silhouette score 0.4785
3. **hybrid-BERTRoBERTa** (BERT-family) with silhouette score 0.4553

### Performance by Category

- **BERT-family**: Average silhouette score 0.4404 (K-means), 0.4127 (GMM)
- **Word2Vec**: Average silhouette score 0.3577 (K-means), 0.3510 (GMM)
- **TF-IDF**: Average silhouette score 0.2133 (K-means), 0.1764 (GMM)
