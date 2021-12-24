import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

df_1950 = pd.read_csv("1950.csv")
df_1960 = pd.read_csv("1960.csv")
df_1970 = pd.read_csv("1970.csv")
df_1980 = pd.read_csv("1980.csv")
df_1990 = pd.read_csv("1990.csv")
df_2000 = pd.read_csv("2000.csv")
df_2010 = pd.read_csv("2010.csv")
df_top10 = pd.read_csv("top10s.csv")

# ubah data has_win_award dari float ke boolean
df_1950 = df_1950.astype({"has_win_award":bool})
df_1960 = df_1960.astype({"has_win_award":bool})
df_1970 = df_1970.astype({"has_win_award":bool})
df_1980 = df_1980.astype({"has_win_award":bool})
df_1990 = df_1990.astype({"has_win_award":bool})
df_2000 = df_2000.astype({"has_win_award":bool})
df_2010 = df_2010.astype({"has_win_award":bool})
df_top10 = df_top10.astype({"has_win_award":bool})

genre_total = pd.unique(df_1950.loc[:,'genre'])
genre_total = pd.unique(np.append(genre_total, pd.unique(df_1960.loc[:,'genre'])))
genre_total = pd.unique(np.append(genre_total, pd.unique(df_1970.loc[:,'genre'])))
genre_total = pd.unique(np.append(genre_total, pd.unique(df_1980.loc[:,'genre'])))
genre_total = pd.unique(np.append(genre_total, pd.unique(df_1990.loc[:,'genre'])))
genre_total = pd.unique(np.append(genre_total, pd.unique(df_2000.loc[:,'genre'])))
genre_total = pd.unique(np.append(genre_total, pd.unique(df_2010.loc[:,'genre'])))

# make scaler first
attribute_spotify_tmp = ["bpm","nrgy","dnce","dB","live","val","dur","acous","spch","popularity","has_win_award"]
attribute_spotify = ["bpm","nrgy","dnce","dB","live","val","acous","spch","popularity"]

sc = StandardScaler()
df_top10_std = sc.fit_transform(df_top10.loc[:,attribute_spotify])
df_top10_std = pd.DataFrame(df_top10_std, columns=attribute_spotify)

# menggunakan clustering kmeans
# mencari nilai optimal dari k
sum_of_squared_distance = []
k = range(1,1000)
for i in k:
    cluster_data = KMeans(n_clusters=i)
    cluster_data = cluster_data.fit(df_top10_std)
    sum_of_squared_distance.append(cluster_data.inertia_)

plt.plot(k,sum_of_squared_distance, "bx-")
plt.xlabel="K"
plt.ylabel="sum of squared distance"
plt.title="optimal of k"
plt.show()