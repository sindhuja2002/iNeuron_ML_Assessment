import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("data/train.csv", index_col=[0])

nltk.download('stopwords')
nltk.download('punkt')

# Tokenization
data['question1'] = data['question1'].str.lower().apply(nltk.word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['question1'] = data['question1'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Stemming
stemmer = PorterStemmer()
data['question1'] = data['question1'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

# Tokenization
data['question2'] = data['question2'].str.lower().apply(nltk.word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['question2'] = data['question2'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Stemming
stemmer = PorterStemmer()
data['question2'] = data['question2'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

combined = data['question1'] + data['question2']

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(combined.apply(' '.join))

cosine_similarities = cosine_similarity(tfidf_matrix)

given_data_index = 0  # Index of the given data point

most_similar_cosine = cosine_similarities[given_data_index].argsort()[-2]

print("Most similar data using Cosine similarity:", data.iloc[most_similar_cosine])
