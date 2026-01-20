import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    # Sentence tokenization
    sentences = sent_tokenize(text)

    tokenized_sentences = []
    for sent in sentences:
        sent = sent.lower()
        sent = re.sub(r"[^a-z\s]", "", sent)
        words = word_tokenize(sent)
        words = [w for w in words if w not in stop_words]
        tokenized_sentences.append(words)

    return sentences, tokenized_sentences
