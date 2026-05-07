# ── Practical 4: TF-IDF Matrix ────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

documents = [
    "The cat sat on the mat",
    "The dog sat on the log",
    "The cat chased the dog",
    "NLP is a fascinating field of AI",
    "Machine learning drives modern AI applications",
]

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Display as a readable DataFrame
df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out(),
    index=[f"Doc {i+1}" for i in range(len(documents))]
)
print("TF-IDF Matrix:")
print(df.round(3).to_string())

# Top 3 important words per document
print("\nTop keywords per document:")
for i, doc in enumerate(documents):
    row = df.iloc[i]
    top = row.nlargest(3)
    print(f"  Doc {i+1}: {dict(zip(top.index, top.round(3)))}")
