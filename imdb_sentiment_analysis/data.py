import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

DIM = 2048

df = pd.read_csv("IMDB.csv")
df['label'] = 1
df.loc[df['sentiment'] == 'negative', 'label'] = 0
df['review_clean'] = df['review'].str.replace(r'<.*?>', ' ', regex=True)

vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words='english',
    max_features=DIM,
    min_df=5,
    max_df=0.9,
    ngram_range=(1, 2)
)
#X = list(vectorizer.fit_transform(df['review_clean'].tolist()))
X = (vectorizer.fit_transform(df['review_clean']))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, list(df['label']), test_size=0.2, random_state=42)



