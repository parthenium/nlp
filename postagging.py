# ── Practical 6: POS Tagging & Noun Extraction ────────────────────
import nltk
from collections import Counter
nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

text = """
Barack Obama served as the 44th President of the United States.
He was born in Hawaii and later moved to Chicago to work as a lawyer.
Obama signed the Affordable Care Act into law in 2010.
"""

tokens = nltk.word_tokenize(text)
pos_tags = nltk.pos_tag(tokens)

print("POS Tags (first 15 tokens):")
for word, tag in pos_tags[:15]:
    print(f"  {word:<15} {tag}")

# Extract all nouns (NN, NNS, NNP, NNPS)
nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
print("\nNouns found:", nouns)

# POS frequency dictionary
pos_freq = Counter(tag for _, tag in pos_tags)
print("\nPOS frequency:")
for tag, freq in sorted(pos_freq.items(), key=lambda x: -x[1]):
    print(f"  {tag:<10} {freq}")
