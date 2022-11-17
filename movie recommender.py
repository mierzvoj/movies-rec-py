import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

# Łukasz Cettler s20168
# Wojciech Mierzejewski s21617
# Movies Recommendation System

movies_df = pd.read_csv('data/movies1.csv',
                        usecols=['userId', 'movieId', 'title', 'rating'],
                        dtype={'userId': 'int32', 'movieId': 'int32', 'title': 'str', 'rating': 'float32'}, encoding='UTF-8')


df = movies_df

combine_movie_rating = df.dropna(axis=0, subset=['title'])
#Eliminating empty fields in the csv file
movie_ratingCount = (combine_movie_rating.
groupby(by=['title'])['rating'].
count().
reset_index().
rename(columns={'rating': 'totalRatingCount'})
[['title', 'totalRatingCount']]
)
#Transforming data matrix,renaming columns
rating_with_totalRatingCount = combine_movie_rating.merge(movie_ratingCount, left_on='title', right_on='title',
                                                          how='left')

pd.set_option('display.float_format', lambda x: '%.3f' % x)


popularity_threshold = 0
rating_popular_movie = rating_with_totalRatingCount

## First lets create a Pivot matrix
movie_features_df = rating_popular_movie.pivot_table(index='title', columns='userId', values='rating').fillna(0)
# First lets create a Pivot matrix
movie_features_df_matrix = csr_matrix(movie_features_df.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(movie_features_df_matrix)

query_index = np.random.choice(movie_features_df.shape[0])
print(query_index)
distances, indices = model_knn.kneighbors(movie_features_df.iloc[query_index, :].values.reshape(1, -1), n_neighbors=6)

for i in range(0, len(distances.flatten())):
    if i == 0:
        print('Recommendations for user {0}:\n'.format(movie_features_df.columns[0]))
    else:
        print('{0}: {1}, with distance of {2}:'.format(i, movie_features_df.index[indices.flatten()[i]],
                                                       distances.flatten()[i]))
