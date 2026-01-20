import numpy as np
from preprocessing import preprocess_text
from word_embeddings import get_word2vec_embeddings
from sentence_embeddings import get_sentence_bert_embeddings
from contextual_embeddings import get_bert_embeddings
from textrank_summarizer import textrank_summary
from fusion import fuse_embeddings
from evaluation import similarity_score, get_rouge_scores

# 1. Load input text
with open("data\sample_text.txt", "r", encoding="utf-8") as file:
    text = file.read()

# 2. Preprocessing
sentences, tokenized_sentences = preprocess_text(text)
summary_ratio = 0.6  # Reduced from 0.5 for a more concise, "extractive" challenge

# 3. Generate Multi-Stage Embeddings
word_emb = get_word2vec_embeddings(tokenized_sentences)
sent_emb = get_sentence_bert_embeddings(sentences)
bert_emb = get_bert_embeddings(sentences)

# 4. Fuse Embeddings with Normalization
fused_emb = fuse_embeddings(word_emb, sent_emb, bert_emb)

# 5. Generate Summaries using TextRank logic
summary_single = textrank_summary(sentences, sent_emb, summary_ratio=summary_ratio)
summary_fused = textrank_summary(sentences, fused_emb, summary_ratio=summary_ratio)

# 6. Quantitative Evaluation
# We use the raw S-BERT embeddings as the 'Semantic Ground Truth' 
# to compare which summary captures the overall meaning better.
original_semantic_center = np.mean(sent_emb, axis=0)

# Helper to get the semantic center of selected sentences
def get_summary_center(full_sentences, summary_text, embeddings):
    # Identify which original indices were chosen for the summary
    indices = [i for i, s in enumerate(full_sentences) if s in summary_text]
    return np.mean(embeddings[indices], axis=0)

center_single = get_summary_center(sentences, summary_single, sent_emb)
center_fused = get_summary_center(sentences, summary_fused, sent_emb)

# Calculate Scores
sim_single = similarity_score(original_semantic_center, center_single)
sim_fused = similarity_score(original_semantic_center, center_fused)

rouge_single = get_rouge_scores(text, summary_single)
rouge_fused = get_rouge_scores(text, summary_fused)

# 7. Final Presentation
print(text)
print(len(text))
print("\n" + "="*60)
print("     MULTI-STAGE EMBEDDING FUSION: EXPERIMENTAL RESULTS")
print("="*60)
print(f"{'Metric':<25} | {'S-BERT (Single)':<15} | {'Fused (Proposed)':<15}")
print("-" * 60)
print(f"{'Semantic Similarity':<25} | {sim_single:.4f}          | {sim_fused:.4f}")
print(f"{'ROUGE-1 (F1-Score)':<25} | {rouge_single['rouge1'].fmeasure:.4f}          | {rouge_fused['rouge1'].fmeasure:.4f}")
print(f"{'ROUGE-L (F1-Score)':<25} | {rouge_single['rougeL'].fmeasure:.4f}          | {rouge_fused['rougeL'].fmeasure:.4f}")
print("-" * 60)

print("\n--- FINAL FUSED SUMMARY ---")
print(summary_fused)
print(len(summary_fused))