import kmapper as km
import numpy as np

from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import Isomap
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_20newsgroups

newsgroups = fetch_20newsgroups(subset='train')
X, y, target_names = np.array(newsgroups.data[:1000]), np.array(newsgroups.target[:1000]), np.array(newsgroups.target_names[:1000])
print("SAMPLE", X[0])
print("SHAPE", X.shape)
print("TARGET", target_names[y[0]])


mapper = km.KeplerMapper(verbose=2)

projected_X = mapper.fit_transform(X,
    projection=[TfidfVectorizer(analyzer="char",
                                ngram_range=(1,6),
                                max_df=0.83,
                                min_df=0.05),
                TruncatedSVD(n_components=100,
                             random_state=1729),
                Isomap(n_components=2,
                       n_jobs=-1)],
    scaler=[None, None, MinMaxScaler()])


print("SHAPE",projected_X.shape)
