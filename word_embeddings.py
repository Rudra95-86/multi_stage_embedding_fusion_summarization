from gensim.models import Word2Vec
import numpy as np

def get_word2vec_embeddings(tokenized_sentences):
    """
    Train Word2Vec and generate sentence embeddings
    Handles empty sentences safely
    """

    # Train Word2Vec on all non-empty sentences
    non_empty_sentences = [s for s in tokenized_sentences if len(s) > 0]

    model = Word2Vec(
        sentences=non_empty_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4
    )

    sentence_vectors = []

    for sentence in tokenized_sentences:
        if len(sentence) == 0:
            # If sentence is empty, use zero vector
            sentence_vectors.append(np.zeros(100))
        else:
            vectors = [model.wv[word] for word in sentence]
            sentence_vectors.append(np.mean(vectors, axis=0))

    return np.vstack(sentence_vectors)
