import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import folium

data = pd.read_csv(r"ML/data-large/rideshhare_kaggle.csv", index_col=[0])

data = data.dropna()

locations = data[['latitude','longitude']]

kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(locations)

X = data[['latitude', 'longitude']]  # Features
y = data['price'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


map = folium.Map(location=[data['latitude'].mean(), data['longitude'].mean()], zoom_start=12)

# Add markers for high booking areas
for i in range(5):
    cluster_locations = locations[clusters == i]
    for _, location in cluster_locations.iterrows():
        folium.Marker([location['latitude'], location['longitude']], icon=folium.Icon(color='red')).add_to(map)

# Add color-coded markers for predicted prices
for _, location in data.iterrows():
    price = model.predict([[location['latitude'], location['longitude']]])
    folium.CircleMarker([location['latitude'], location['longitude']],
                        radius=5,
                        color='blue',
                        fill_color='blue',
                        fill_opacity=0.7,
                        popup=f"Price: ${price[0]:.2f}").add_to(map)

# Save the map
map.save('map.html')