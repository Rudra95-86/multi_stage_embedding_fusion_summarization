import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# --- UPDATED DOWNLOAD LOGIC FOR CLOUD DEPLOYMENT ---
try:
    nltk.data.find('tokenizers/punkt_tab')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Explicitly download the missing resource
    nltk.download('stopwords')
# --------------------------------------------------

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
