from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rouge_score import rouge_scorer

def similarity_score(text_embedding, summary_embedding):
    """
    Computes semantic similarity between the document and its summary.
    """
    # If summary is multiple sentences, take the mean to represent the summary vector
    if len(summary_embedding.shape) > 1:
        summary_embedding = np.mean(summary_embedding, axis=0)
    
    # If original text is multiple sentences, take the mean
    if len(text_embedding.shape) > 1:
        text_embedding = np.mean(text_embedding, axis=0)

    score = cosine_similarity(
        text_embedding.reshape(1, -1),
        summary_embedding.reshape(1, -1)
    )
    return score[0][0]

def get_rouge_scores(reference_text, summary_text):
    """
    Computes ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference_text, summary_text)
    return scores