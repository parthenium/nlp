# ── Practical 1: Text Pre-processing ──────────────────────────────
import nltk
nltk.download('punkt'); nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

import string, re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = """Hello! This is a Sample TEXT for NLP Pre-processing.
          Running, runs, ran — all forms of 'run'. Visit https://example.com"""

# 1. Tokenize
tokens = word_tokenize(text)
print("Tokens:", tokens)

# 2. Lowercase
tokens = [t.lower() for t in tokens]

# 3. Remove punctuation & special chars
tokens = [re.sub(r'[^a-z]', '', t) for t in tokens]
tokens = [t for t in tokens if t]  # remove empty strings

# 4. Remove stopwords
stop_words = set(stopwords.words('english'))
tokens = [t for t in tokens if t not in stop_words]
print("After stopword removal:", tokens)

# 5. Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(t) for t in tokens]
print("Stemmed:", stemmed)

# 6. Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(t) for t in tokens]
print("Lemmatized:", lemmatized)
