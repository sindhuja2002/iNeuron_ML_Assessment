import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, manhattan_distances

# Step 1: Load the dataset
data = pd.read_json(r"data\News_Category_Dataset_v3.json", lines = True) 

# Step 2: Preprocess the text data
nltk.download('stopwords')
nltk.download('punkt')

# Tokenization
data['processed_text'] = data['headline'].str.lower().apply(nltk.word_tokenize)

# Remove stopwords
stop_words = set(stopwords.words('english'))
data['processed_text'] = data['processed_text'].apply(lambda tokens: [word for word in tokens if word not in stop_words])

# Stemming
stemmer = PorterStemmer()
data['processed_text'] = data['processed_text'].apply(lambda tokens: [stemmer.stem(word) for word in tokens])

# Step 3: Compute similarity scores
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['processed_text'].apply(' '.join))
cosine_similarities = cosine_similarity(tfidf_matrix)
euclidean_distances = euclidean_distances(tfidf_matrix)
manhattan_distances = manhattan_distances(tfidf_matrix)

# Compute Jaccard similarities (for categorical data)
jaccard_similarities = pd.DataFrame(index=data.index, columns=data.index)
for i in data.index:
    for j in data.index:
        intersection = set(data.loc[i]) & set(data.loc[j])
        union = set(data.loc[i]) | set(data.loc[j])
        jaccard_similarities.loc[i, j] = len(intersection) / len(union)

# Step 4: Find the most similar data
given_data_index = 0  # Index of the given data point

most_similar_cosine = cosine_similarities[given_data_index].argsort()[-2]
most_similar_euclidean = euclidean_distances[given_data_index].argsort()[1]
most_similar_manhattan = manhattan_distances[given_data_index].argsort()[1]
most_similar_jaccard = jaccard_similarities[given_data_index].drop(given_data_index).idxmax()

# Print the most similar data
print("Most similar data using Cosine similarity:", data.iloc[most_similar_cosine])
print("Most similar data using Euclidean distance:", data.iloc[most_similar_euclidean])
print("Most similar data using Manhattan distance:", data.iloc[most_similar_manhattan])
print("Most similar data using Jaccard similarity:", data.iloc[most_similar_jaccard])
