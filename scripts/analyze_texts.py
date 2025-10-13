import pandas as pd
from collections import Counter
import re

import nltk
nltk.download("stopwords")

from nltk.corpus import stopwords


df = pd.read_csv('data/messages.csv')
texts = ' '.join(df['text'].astype(str)).lower()
words = re.findall(r'\b[а-яa-z]+\b', texts)

russian_stopwords = set(stopwords.words('russian'))
filtered_words = [w for w in words if w not in russian_stopwords and len(w) > 2]

counter = Counter(filtered_words)
print(counter.most_common(20))