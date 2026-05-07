# ── Practical 3: Word Frequency (excluding stopwords) ─────────────
import nltk, re
from collections import Counter
nltk.download('stopwords'); nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science,
and artificial intelligence concerned with the interactions between computers and
human language, in particular how to program computers to process and analyze
large amounts of natural language data.
"""

stop_words = set(stopwords.words('english'))

tokens = word_tokenize(text.lower())
tokens = [re.sub(r'[^a-z]', '', t) for t in tokens]
tokens = [t for t in tokens if t and t not in stop_words]

freq = Counter(tokens)
print("Top 10 most common words:")
for word, count in freq.most_common(10):
    print(f"  {word:<20} {count}")
