# ── Practical 5: Unigram & Bigram Language Model ──────────────────
import nltk, re, math
from collections import Counter, defaultdict
nltk.download('punkt')
from nltk.tokenize import word_tokenize

corpus = """
the cat sat on the mat the cat is fat
the dog sat on the log the dog is big
cats and dogs are pets pets are fun
"""

tokens = word_tokenize(corpus.lower())
tokens = [t for t in tokens if re.match(r'^[a-z]+$', t)]

V = len(set(tokens))   # vocabulary size
N = len(tokens)

# ── Unigram with add-1 (Laplace) smoothing
unigram_counts = Counter(tokens)
def unigram_prob(word):
    return (unigram_counts[word] + 1) / (N + V)

# ── Bigram with add-1 smoothing
bigrams = list(zip(tokens[:-1], tokens[1:]))
bigram_counts = Counter(bigrams)

def bigram_prob(w1, w2):
    return (bigram_counts[(w1, w2)] + 1) / (unigram_counts[w1] + V)

# ── Sentence probability
def sentence_prob(sentence):
    words = word_tokenize(sentence.lower())
    log_p = 0
    for i, w in enumerate(words):
        if i == 0:
            log_p += math.log(unigram_prob(w))
        else:
            log_p += math.log(bigram_prob(words[i-1], w))
    return math.exp(log_p)

test_sentences = ["the cat sat on the mat", "dog is big", "fish fly high"]
print("\nSentence Probabilities:")
for s in test_sentences:
    p = sentence_prob(s)
    print(f"  '{s}'")
    print(f"    Probability : {p:.10f}")
  
  
