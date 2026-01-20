from sentence_transformers import SentenceTransformer

# Load model once
model = SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_sentence_bert_embeddings(sentences):
    """
    Returns real sentence embeddings
    Shape: (num_sentences, 384)
    """
    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=False
    )
    return embeddings

