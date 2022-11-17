import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

movies_df = pd.read_csv('data/movies1.csv',
                        usecols=['userId', 'movieId', 'title', 'rating'],
                        dtype={'userId': 'int32', 'movieId': 'int32', 'title': 'str', 'rating': 'float32'}, encoding='UTF-8')


movie_features_df = movies_df.pivot_table(index='title', columns='userId', values='rating').fillna(0)

movie_features_df_matrix = csr_matrix(movie_features_df.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_features_df_matrix)

query_index = np.random.choice(movie_features_df.shape[0])

distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for user {0}:\n'.format(movie_features_df.columns[0]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]],
                                                       distances.flatten()[i]))