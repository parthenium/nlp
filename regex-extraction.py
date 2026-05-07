# ── Practical 2: Regex Pattern Extraction ────────────────────────
import re

text = """
Contact us at support@example.com or admin@test.org
Tweet us @john_doe or @jane123
Meeting on 2024-03-15 or 15/03/2024 or March 15, 2024
Call +91-9876543210 or (022) 1234-5678 or 9988776655
"""

# 1. Email usernames (part before @)
emails = re.findall(r'([a-zA-Z0-9_.+-]+)@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
print("Email usernames:", emails)

# 2. Twitter / social media handles
handles = re.findall(r'(?<!\S)@(\w+)', text)
print("Handles:", handles)

# 3. Dates — multiple formats
dates = re.findall(
    r'\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|[A-Z][a-z]+ \d{1,2}, \d{4})\b', text)
print("Dates:", dates)

# 4. Phone numbers
phones = re.findall(r'(\+?\d{1,3}[-.\s]?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4})', text)
print("Phones:", phones)
