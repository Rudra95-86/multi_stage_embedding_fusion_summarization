import numpy as np
from sklearn.preprocessing import StandardScaler

# Improvement for fusion.py
def fuse_embeddings(word_emb, sent_emb, bert_emb):
    scaler = StandardScaler()
    
    # Weights: Give S-BERT and BERT more weight than Word2Vec
    # to emphasize context over simple word-level semantics
    fused = np.concatenate((
        scaler.fit_transform(word_emb) * 0.5, # Local semantics
        scaler.fit_transform(sent_emb) * 1.2, # Sentence context
        scaler.fit_transform(bert_emb) * 1.0  # Thematic info
    ), axis=1)
    return fused