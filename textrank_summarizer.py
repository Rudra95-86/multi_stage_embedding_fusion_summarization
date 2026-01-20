from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def textrank_summary(sentences, embeddings, summary_ratio=0.25, redundancy_threshold=0.8):
    """
    Dynamic extractive summarizer using embeddings and TextRank-style scoring.
    
    Parameters:
    - sentences: list of original sentences
    - embeddings: sentence embeddings (Sentence-BERT or fused)
    - summary_ratio: fraction of sentences to include (default 0.25)
    - redundancy_threshold: max cosine similarity between selected sentences
    
    Returns:
    - summary: string of selected sentences in original order
    """

    num_sentences = max(1, int(len(sentences) * summary_ratio))

    # Compute similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # Score sentences by sum of similarities (TextRank approximation)
    scores = sim_matrix.sum(axis=1)

    # Rank sentences by score
    ranked_indices = np.argsort(scores)[::-1]

    # Select sentences dynamically with redundancy check
    selected = []

    for idx in ranked_indices:
        # Skip sentence if too similar to already selected sentences
        if all(cosine_similarity(embeddings[idx].reshape(1, -1), embeddings[s].reshape(1, -1))[0][0] < redundancy_threshold for s in selected):
            selected.append(idx)
        if len(selected) >= num_sentences:
            break

    # Ensure first and last sentences are included (intro + moral)
    if 0 not in selected:
        selected.append(0)
    if len(sentences) > 1 and (len(sentences) - 1) not in selected:
        selected.append(len(sentences) - 1)

    # Keep chronological order
    selected = sorted(selected)

    # Return summary text
    summary = " ".join([sentences[i] for i in selected])
    return summary
