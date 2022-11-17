import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Łukasz Cettler s20168
# Wojciech Mierzejewski s21617
# Movies Recommendation System

#Data input from CSV file 
movies_df = pd.read_csv('data/movies1.csv',
                        usecols=['userId', 'movieId', 'title', 'rating'],
                        encoding='utf-8')

#Creating Pivot table, replacing NaN values with 0
movie_features_df = movies_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)
#Transformation movie_features_df matrix into csr sparse matrix, to store non zero values and save memory
movie_features_df_matrix = csr_matrix(movie_features_df.values)
#Nearest neighbours model implements cosine similarity distance comparison
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
#Training model
model_knn.fit(movie_features_df_matrix)
#Generating random row index
query_index = np.random.choice(movie_features_df.shape[0])
#Finding nearest neighbours
distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)
#Loop over array flattened to single dimension
for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for user {0}:\n'.format(movie_features_df.columns[0]))
        print("--------------")
    else:
        print(",,,,,,,,,,,,,,")
        print('{0}: {1}'.format(i, movie_features_df.index[indices.flatten()[i]]))
                                                     