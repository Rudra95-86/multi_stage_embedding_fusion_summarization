import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def extractive_summary(sentences, embeddings, num_sentences=4):
    """
    Improved extractive summarization:
    """
    similarity_matrix = cosine_similarity(embeddings)
    scores = similarity_matrix.sum(axis=1)
    ranked = np.argsort(scores)[::-1]

    selected = set()

    selected.add(0)

    mid_index = len(sentences) // 2
    selected.add(mid_index)

    for idx in ranked:
        if len(selected) < num_sentences:
            selected.add(idx)

    # Keep chronological order
    selected = sorted(list(selected))[:num_sentences]

    summary = [sentences[i] for i in selected]
    return " ".join(summary)
