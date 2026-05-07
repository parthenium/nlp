# ── Practical 7: Named Entity Recognition (NER) ───────────────────
# Run this cell first to download the model:
# !python -m spacy download en_core_web_sm

import spacy

nlp = spacy.load("en_core_web_sm")

headlines = [
    "Apple acquires British AI startup for $200 million",
    "Elon Musk visits Tokyo to meet with Toyota executives",
    "NASA launches Artemis mission from Cape Canaveral on Friday",
    "The Federal Reserve raises interest rates by 25 basis points",
    "Manchester United signs Brazilian midfielder from Flamengo",
]

for headline in headlines:
    doc = nlp(headline)
    print(f"\nHeadline: {headline}")
    if doc.ents:
        for ent in doc.ents:
            print(f"  {ent.text:<30} → {ent.label_} ({spacy.explain(ent.label_)})")
    else:
        print("  No named entities found")
