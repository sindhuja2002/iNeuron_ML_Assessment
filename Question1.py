import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np

data = pd.read_csv("Mlnotebooks/data/instagram_reach.csv", index_col=[0])
label = LabelEncoder()
caption_vectorizer = CountVectorizer()
hasht_vectorizer = CountVectorizer()
pca = PCA(n_components=50)
likes_model = LinearRegression()
time_model = LinearRegression()


data = data.dropna()

# Split the data into input features (X) and target variables (y)
X = data.drop(['Likes', 'Time since posted'], axis=1)  # Remove the target variables from the input features
y_likes = data['Likes']  # Target variable: number of likes
y_time_since_posted = data['Time since posted'].str.split().str[0].astype(int)  # Target variable: time since posted

X = X.reset_index( drop = True)


X_caption = caption_vectorizer.fit_transform(X['Caption'])
X_caption = pd.DataFrame(X_caption.toarray(), columns=caption_vectorizer.get_feature_names_out())
X.drop('Caption', axis=1, inplace=True)

X = pd.concat([X, X_caption], axis=1, join = 'inner')


X_hash = hasht_vectorizer.fit_transform(X['Hashtags'])
X_hash = pd.DataFrame(X_hash.toarray(), columns=hasht_vectorizer.get_feature_names_out())
X.drop('Hashtags', axis=1, inplace=True)

X = pd.concat([X, X_hash], axis = 1, join = 'inner')

X['username'] = label.fit_transform(X['USERNAME'])

X =X.drop(['USERNAME'], axis = 1)

X_pca = pca.fit_transform(X)

X_train, X_test, y_likes_train, y_likes_test, y_time_train, y_time_test = train_test_split(
    X_pca, y_likes, y_time_since_posted, test_size=0.2, random_state=42
)

likes_model.fit(X_train, y_likes_train)

likes_predictions = likes_model.predict(X_test)
likes_predictions = likes_predictions.round().astype(int)

mse_likes = mean_squared_error(y_likes_test, likes_predictions)
mae_likes = mean_absolute_error(y_likes_test, likes_predictions)
print("Mean Squared Error (Likes):", mse_likes)
print("Mean Absolute Error (Likes):", mae_likes)

time_model.fit(X_train, y_time_train)

time_predictions = time_model.predict(X_test)
time_predictions = time_predictions.round().astype(int)
time_pred = np.array([str(int(pred)) + ' hours' for pred in time_predictions])

mae_time = mean_absolute_error(y_time_test, time_predictions)
rmse_time = mean_squared_error(y_time_test, time_predictions, squared=False)
print("Mean Absolute Error (Time Since Posted):", mae_time)
print("Root Mean Squared Error (Time Since Posted):", rmse_time)