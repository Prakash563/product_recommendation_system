

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from google.colab import drive
drive.mount('/content/drive')

df=pd.read_csv('/content/drive/MyDrive/ratings.csv',header=None)
#as there is no proper header , we are going to name the columns later
df.head()

df.columns=['userid','productid','ratings','timestamp']
print('overall info of the dataset')
print('---------------------------')
print(df.info())
print(df.head())

#checking the nullvalues
df.isnull().sum()

df.duplicated().sum()#checking the duplicated values

print('unique users of the data: ',df['userid'].nunique())
print("unique products present in data:",df['productid'].nunique())

print('overall shape of the data:')
print('--------------------------')
print(df.shape)

"""##### as you can see there are no null values are present in the data
* we dont require the timestamp dataset, so we are dropping the timestamp column completely
"""

df.drop(columns='timestamp',inplace=True)

df.head()#after removing the timestamp column

df["ratings"].value_counts().sort_index()#it typically shows how many 1,2,3,4 and 5 stars ratings is given by how many users on how  it is shown

#VISUALISATION OF NO. OF RATINGS
plt.figure(figsize=(6, 4))
ax=sns.countplot(data=df, x="ratings")#it shows the ratings in the barchart format
plt.title("Rating Distribution")
plt.xlabel("Ratings")
plt.ylabel("Number of Ratings")
for container in ax.containers:
    labels = [f"{int(bar.get_height()):,}" for bar in container]
    ax.bar_label(container, labels=labels, padding=3)
plt.show()

#getting the top 10 most rated products
top_prod=df['productid'].value_counts(ascending=False).head(10)
top_prod

#VISUALIZATION OF TOP 10 RATED PRODUCTS
plt.figure(figsize=(6,4))
ax=top_prod.plot(kind='bar')
ax.bar_label(ax.containers[0], fmt="%d")
plt.title('top10 most rated products')
plt.xlabel("product id")
plt.ylabel('No. of ratings')
plt.xticks(rotation=45)
plt.show()

#getting the most users who have bought the products
top_users=df['userid'].value_counts().head(10)
top_users

#VISUALIZATION OF TOP 10 USERS WHO BOUGHT MORE PRODUCTS
plt.figure(figsize=(6,4))
rx=top_users.plot(kind='bar')
rx.bar_label(rx.containers[0], fmt="%d")
plt.title('top10 users who brought more products')
plt.xlabel("product id")
plt.ylabel('No. of ratings')
plt.xticks(rotation=45)
plt.show()

#CHECKING THE NO.OF RATINGS GIVEN BY EVERY USER SUMMARY
ratings_per_user=df.groupby('userid')['ratings'].count()
print('------------------------')
print('ratings per user summary')
print("------------------------")
ratings_per_user.describe().apply(lambda x: f"{x:,.2f}")

#Ratings Per Product Summary
ratings_per_product = df.groupby("productid")["ratings"].count()
print('----------------------------')
print('ratings_per_product_summary')
print("----------------------------")
ratings_per_product.describe()

"""lets talk about the term **DATA SPARSITY**
* Data sparsity is a condition where a dataset has a large number of missing or zero-valued entries relative to the total possible entries.
* in short the formula can be mentioned as :

$$
\text{Data Sparsity} =
\frac{\text{Number of zero/missing values}}
{\text{Total number of values}}
\times 100\%
$$



"""

#checking dataset sparsity in the dataset
total_possible_ratings = df["userid"].nunique() * df["productid"].nunique()
actual_ratings = len(df)
sparsity = (1- actual_ratings / total_possible_ratings) * 100
print("Sparsity:", round(sparsity, 4), "%")
#it clearly shows that 99.99 percent ratings are missing that every user is not rating each product, this is usual in the dataset like not every user can able to buy every product and make to give a review

n_users = df['userid'].nunique()
n_products = df['productid'].nunique()
print(f"User-item matrix would be {n_users:,} x {n_products:,} = {n_users*n_products:,.0f} cells")
print(f"Only {actual_ratings:,} are filled -> {sparsity:.4f}% sparse")
print()
print(f"Users with exactly 1 rating: {(ratings_per_user==1).sum():,} ({(ratings_per_user==1).mean()*100:.1f}%)")
print()
print(f"Products with exactly 1 rating: {(ratings_per_product==1).sum():,} ({(ratings_per_product==1).mean()*100:.1f}%)")

df.describe()#description of the dataset

from scipy.sparse import csr_matrix

##JUST SHOWING THE SAMPLE OF THE SPARSITY DATASET WHICH IN SMALL EXAMPLE THAT EVERY CANT BE ABLE TO RATE EVERY PRODUCT, THE MISSING BOXES IN THE DATA
#REPRESENTS THE 'MISSING RATINGS' BY THE USER TO  THE GIVEN PRODUCT
sample_users = df["userid"].value_counts().head(30).index
sample_products = df["productid"].value_counts().head(30).index

small_matrix = df[
    df["userid"].isin(sample_users) &
    df["productid"].isin(sample_products)
].pivot_table(
    index="userid",
    columns="productid",
    values="ratings"
)

plt.figure(figsize=(12, 7))

sns.heatmap(
    small_matrix.notnull(),
    cmap="Blues",
    cbar=False,
    linewidths=0.5
)

plt.title("Sparse User-Product Matrix")
plt.xlabel("Products")
plt.ylabel("Users")

plt.show()



"""# Model Building & Evaluation #"""

#IMPORTING REQUIRED LIBRARIES FOR THE MODEL BUILDING
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score,davies_bouldin_score
from sklearn.cluster import KMeans,MiniBatchKMeans,dbscan,AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD



#FILTERING ACTIVE USERS AND PRODUCTS

user_counts = df["userid"].value_counts()
product_counts = df["productid"].value_counts()

active_users = user_counts[user_counts >= 10].index
active_products = product_counts[product_counts >= 10].index

model_df = df[
    df["userid"].isin(active_users) &
    df["productid"].isin(active_products)
].copy()

print("Original data shape:", df.shape)
print("Filtered data shape:", model_df.shape)

"""recommendation systems need enough rating history. Users with very few ratings and products with few ratings wont provide strong patterns. so i kept users with atleast 10 ratings"""

model_df["user_index"] = model_df["userid"].astype("category").cat.codes
model_df["product_index"] = model_df["productid"].astype("category").cat.codes

user_mapping = model_df[["userid", "user_index"]].drop_duplicates()
product_mapping = model_df[["productid", "product_index"]].drop_duplicates()

model_df.head()

"""i have given the userid as unique index value because the ml models cannot understand the text ids , so i have converted the user and product ids into numerical indexes"""

user_product_matrix = csr_matrix(
    (
        model_df["ratings"],
        (model_df["user_index"], model_df["product_index"])
    )
)

user_product_matrix

"""I created a user-product matrix. Rows represent users, columns represent products, and values represent ratings. Since most users rate only a few products, I used a sparse matrix to save memory."""

import matplotlib.pyplot as plt
import seaborn as sns

sample_users = model_df["userid"].value_counts().head(30).index
sample_products = model_df["productid"].value_counts().head(30).index

small_df = model_df[
    model_df["userid"].isin(sample_users) &
    model_df["productid"].isin(sample_products)
]

small_matrix = small_df.pivot_table(
    index="userid",
    columns="productid",
    values="ratings"
)

plt.figure(figsize=(12, 7))

sns.heatmap(
    small_matrix.notnull(),
    cmap="Blues",
    cbar=False,
    linewidths=0.5
)

plt.title("Sparse User-Product Matrix")
plt.xlabel("Product ID")
plt.ylabel("User ID")

plt.show()

"""* blue cell= user rated that product
* white cell= user not rated that product

popularity based model
"""

popular_products = model_df.groupby("productid").agg(
    average_rating=("ratings", "mean"),
    rating_count=("ratings", "count")
).reset_index()

popular_products = popular_products[
    popular_products["rating_count"] >= 50
].sort_values(
    by=["average_rating", "rating_count"],
    ascending=False
)

popular_products.head(10)

"""This is the baseline model. It recommends products that have high average ratings and many ratings. It is not personalized, but it is useful for new users."""

item_model = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=6
)

item_model.fit(user_product_matrix.T)

"""I trained an item-item collaborative filtering model. I used transpose .T because I want to compare products with products. Cosine similarity is used to find products with similar rating patterns."""

def recommend_similar_products(product_id, n=5):
    product_row = product_mapping[
        product_mapping["productid"] == product_id
    ]

    if product_row.empty:
        return "Product not found"

    product_index = product_row["product_index"].values[0]

    distances, indices = item_model.kneighbors(
        user_product_matrix.T[product_index],
        n_neighbors=n + 1
    )

    recommended_indices = indices.flatten()[1:]
    similarity_scores = 1 - distances.flatten()[1:]

    result = pd.DataFrame({
        "product_index": recommended_indices,
        "similarity_score": similarity_scores
    })

    result = result.merge(product_mapping, on="product_index", how="left")

    return result[["productid", "similarity_score"]]

"""This function takes a product ID and recommends similar products. A similarity score closer to 1 means the products are more similar."""

sample_product = model_df["productid"].iloc[0]

similar_products = recommend_similar_products(sample_product, 10)

similar_products

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=similar_products,
    x="similarity_score",
    y="productid",
    color="skyblue"
)

plt.title(f"Cosine Similarity Products for Product {sample_product}")
plt.xlabel("Cosine Similarity Score")
plt.ylabel("Recommended Product ID")
plt.xlim(0, 1)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", padding=3)

plt.show()

#Testing the recommendation model
sample_product = model_df["productid"].iloc[0]

print("Selected product:", sample_product)

recommend_similar_products(sample_product, 5)

"""tested the model by selecting one product and getting five similar products based on user rating behavior."""

def filter_min_ratings(data, min_user_ratings=5, min_product_ratings=10, max_rounds=5):

    d = data.copy()
    for i in range(max_rounds):
        user_counts = d['userid'].value_counts()
        product_counts = d['productid'].value_counts()
        keep_users = user_counts[user_counts >= min_user_ratings].index
        keep_products = product_counts[product_counts >= min_product_ratings].index

        before = len(d)
        d = d[d['userid'].isin(keep_users) & d['productid'].isin(keep_products)]
        after = len(d)

        print(f"  round {i+1}: {before:,} -> {after:,} ratings | "
              f"{d['userid'].nunique():,} users | {d['productid'].nunique():,} products")
        if before == after:
            print("  converged")
            break
    return d

MIN_USER_RATINGS = 5
MIN_PRODUCT_RATINGS = 10

print(f"Filtering to users/products with >= {MIN_USER_RATINGS} ratings each:")
df_filtered = filter_min_ratings(df, MIN_USER_RATINGS, MIN_PRODUCT_RATINGS)

retained_pct = len(df_filtered) / len(df) * 100
print(f"\nRetained {len(df_filtered):,} of {len(df):,} ratings ({retained_pct:.1f}%)")

""""We filtered out users and products with fewer than 10 ratings because they don't have enough data for the model to detect a pattern from, and even though that removes about around 95% of the rows, it's mostly rows that couldn't help the model anyway — and we kept the threshold as a named variable at the top of the notebook so it's easy to adjust in one place during the deployment phase."

Iteratively drop users/products below the ratings threshold.

    This has to be iterative: removing sparse products can drop a user
    below the threshold too (and vice versa), so one pass isn't enough.
    We cap at `max_rounds` rather than running to full convergence —
    convergence keeps shrinking the dataset for many more rounds with
    rapidly diminishing returns.
"""

#DIMENSIONALITY REDUCTION USING TRUNCATEDSVD
svd = TruncatedSVD(n_components=50, random_state=42)

product_features = svd.fit_transform(user_product_matrix.T)

print("Product feature shape:", product_features.shape)
print("Explained variance:", round(svd.explained_variance_ratio_.sum() * 100, 2), "%")

def build_sparse_matrix(data):
    """Build a CSR user-item ratings matrix plus the id<->index lookup tables."""
    user_ids = data['userid'].unique()
    product_ids = data['productid'].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    product_to_idx = {p: i for i, p in enumerate(product_ids)}

    row_idx = data['userid'].map(user_to_idx).values
    col_idx = data['productid'].map(product_to_idx).values
    values = data['ratings'].values

    matrix = csr_matrix((values, (row_idx, col_idx)),
                         shape=(len(user_ids), len(product_ids)))
    return matrix, user_to_idx, product_to_idx, user_ids, product_ids

user_product_matrix, user_to_idx, product_to_idx, user_ids, product_ids = build_sparse_matrix(df_filtered)

density = user_product_matrix.nnz / (user_product_matrix.shape[0] * user_product_matrix.shape[1])
print(f"Matrix shape: {user_product_matrix.shape}  |  non-zero entries: {user_product_matrix.nnz:,}  |  density: {density*100:.3f}%")

"""user_to_idx` / `product_to_idx` matter beyond just building the matrix — we reuse them every time we need to go from a real `userId`/`productId` back to a row/column number, including in the recommendation function in . Keeping the mappings as separate, reusable objects (instead of re-deriving them inline each time) is what keeps the rest of the notebook from silently going out of sync with itself."""

def get_svd_embeddings(matrix, n_components=50, random_state=42):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    embeddings = svd.fit_transform(matrix)
    return svd, embeddings

svd_model, user_embeddings = get_svd_embeddings(user_product_matrix, n_components=50)

print(f"Embeddings shape: {user_embeddings.shape}")
print(f"Variance explained by 50 components: {svd_model.explained_variance_ratio_.sum()*100:.1f}%")

"""The product matrix is very large and sparse. So I used TruncatedSVD to reduce it into 50 important features. This makes clustering faster and easier"""

# Elbow method: how does K-Means inertia change as we add more clusters?
# MiniBatchKMeans is used here purely for speed while scanning k.
k_range = range(2, 11)
inertias = []

for k in k_range:
    mbk = MiniBatchKMeans(n_clusters=k, random_state=42, n_init=5, batch_size=1024)
    mbk.fit(user_embeddings)
    inertias.append(mbk.inertia_)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(list(k_range), inertias, marker='o')
ax.set_xlabel("k (number of clusters)")
ax.set_ylabel("Inertia (within-cluster sum of squares)")
ax.set_title("Elbow method for choosing k")
plt.tight_layout()
plt.show()

N_CLUSTERS = 5

kmeans_model = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
kmeans_labels = kmeans_model.fit_predict(user_embeddings)


kmeans_silhouette = silhouette_score(user_embeddings, kmeans_labels, sample_size=5000, random_state=42)
kmeans_db = davies_bouldin_score(user_embeddings, kmeans_labels)

print("K-Means cluster sizes:", np.bincount(kmeans_labels))
print(f"Silhouette score:     {kmeans_silhouette:.4f}  (range -1 to 1, higher = better separated)")
print(f"Davies-Bouldin index: {kmeans_db:.4f}  (lower = better separated)")

cluster_counts = pd.Series(kmeans_labels).value_counts().sort_index()

cluster_counts

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=cluster_counts.index,
    y=cluster_counts.values,
    color="skyblue"
)

plt.title("Number of Users in Each KMeans Cluster")
plt.xlabel("Cluster Number")
plt.ylabel("Number of Users")

for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3)

plt.show()

"""KMeans groups products into clusters based on similar rating behavior. I used 5 clusters here."""

import time

t0 = time.time()
kmeans_model_timed = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
_ = kmeans_model_timed.fit_predict(user_embeddings)
kmeans_fit_time = time.time() - t0

t0 = time.time()
mbkmeans_model = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10, batch_size=1024)
mbkmeans_labels = mbkmeans_model.fit_predict(user_embeddings)
mbkmeans_fit_time = time.time() - t0

mbkmeans_silhouette = silhouette_score(user_embeddings, mbkmeans_labels, sample_size=5000, random_state=42)
mbkmeans_db = davies_bouldin_score(user_embeddings, mbkmeans_labels)

print("MiniBatchKMeans cluster sizes:", np.bincount(mbkmeans_labels))
print(f"Silhouette score:     {mbkmeans_silhouette:.4f}")
print(f"Davies-Bouldin index: {mbkmeans_db:.4f}")
print(f"\nFit time - full K-Means:      {kmeans_fit_time*1000:.1f} ms")
print(f"Fit time - MiniBatchKMeans:  {mbkmeans_fit_time*1000:.1f} ms")

mbk_cluster_counts = pd.Series(mbkmeans_labels).value_counts().sort_index()

mbk_cluster_counts

plt.figure(figsize=(8, 5))

ax = sns.barplot(
    x=mbk_cluster_counts.index,
    y=mbk_cluster_counts.values,
    color="skyblue"
)

plt.title("Number of Users in Each minibatchKMeans Cluster")
plt.xlabel("Cluster Number")
plt.ylabel("Number of Users")

for container in ax.containers:
    ax.bar_label(container, fmt="%d", padding=3)

plt.show()

"""MiniBatchKMeans is a faster version of KMeans. It is useful for large datasets like this one."""

def build_sparse_matrix(data):
    user_ids = data['userid'].unique()
    product_ids = data['productid'].unique()
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    product_to_idx = {p: i for i, p in enumerate(product_ids)}

    row_idx = data['userid'].map(user_to_idx).values
    col_idx = data['productid'].map(product_to_idx).values
    values = data['ratings'].values

    matrix = csr_matrix((values, (row_idx, col_idx)),
                         shape=(len(user_ids), len(product_ids)))
    return matrix, user_to_idx, product_to_idx, user_ids, product_ids

user_item_matrix, user_to_idx, product_to_idx, user_ids, product_ids = build_sparse_matrix(df_filtered)

density = user_item_matrix.nnz / (user_item_matrix.shape[0] * user_item_matrix.shape[1])
print(f"Matrix shape: {user_item_matrix.shape}  |  non-zero entries: {user_item_matrix.nnz:,}  |  density: {density*100:.3f}%")

def get_svd_embeddings(matrix, n_components=50, random_state=42):
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    embeddings = svd.fit_transform(matrix)
    return svd, embeddings

svd_model, user_embeddings = get_svd_embeddings(user_item_matrix, n_components=50)

print(f"Embeddings shape: {user_embeddings.shape}")
print(f"Variance explained by 50 components: {svd_model.explained_variance_ratio_.sum()*100:.1f}%")

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import normalize

X = user_embeddings.astype("float32", copy=False)
X = normalize(X)  # good for embeddings

print("Embedding shape:", X.shape)
print("Approx RAM:", X.nbytes / 1e9, "GB")

dbscan_model = DBSCAN(
    eps=0.35,          # start much smaller than 3
    min_samples=10,
    metric="euclidean"
)

dbscan_labels = dbscan_model.fit_predict(X)

n_dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = int((dbscan_labels == -1).sum())
noise_pct = n_noise / len(dbscan_labels) * 100

print(f"DBSCAN found {n_dbscan_clusters} clusters, {n_noise:,} noise points ({noise_pct:.1f}% of users)")

mask = dbscan_labels != -1
clustered_labels = dbscan_labels[mask]

if len(set(clustered_labels)) >= 2:
    rng = np.random.default_rng(42)
    clustered_idx = np.flatnonzero(mask)
    score_idx = rng.choice(clustered_idx, size=min(5000, len(clustered_idx)), replace=False)

    dbscan_silhouette = silhouette_score(X[score_idx], dbscan_labels[score_idx])
    dbscan_db = davies_bouldin_score(X[score_idx], dbscan_labels[score_idx])

    print(f"Silhouette score excluding noise: {dbscan_silhouette:.4f}")
    print(f"Davies-Bouldin index excluding noise: {dbscan_db:.4f}")
else:
    print("Not enough non-noise clusters to compute silhouette/Davies-Bouldin.")

dbscan_cluster_counts = pd.Series(dbscan_labels).value_counts().sort_index()

dbscan_cluster_counts

plt.figure(figsize=(10, 5))

ax = sns.barplot(
    x=dbscan_cluster_counts.index.astype(str),
    y=dbscan_cluster_counts.values,
    color="salmon"
)

plt.title("Number of Users in Each DBSCAN Cluster")
plt.xlabel("Cluster Label (-1 means Noise)")
plt.ylabel("Number of Users")

for container in ax.containers:
  ax.bar_label(container, fmt="%d", padding=3)

plt.show()

# so we compare it on a representative random subsample. This is a valid comparison
# method (same embedding space, same k, same metrics) but not a substitute for the
# full-data K-Means score above - noted explicitly so it isn't misread as an apples-to-apples number.
rng = np.random.RandomState()
sample_idx = rng.choice(user_embeddings.shape[0], size=3000, replace=False)
embeddings_sample = user_embeddings[sample_idx]

agglo_model = AgglomerativeClustering(n_clusters=N_CLUSTERS)
agglo_labels = agglo_model.fit_predict(embeddings_sample)

agglo_silhouette = silhouette_score(embeddings_sample, agglo_labels)
agglo_db = davies_bouldin_score(embeddings_sample, agglo_labels)

print("Agglomerative cluster sizes (on 3,000-user sample):", np.bincount(agglo_labels))
print(f"Silhouette score:     {agglo_silhouette:.4f}")
print(f"Davies-Bouldin index: {agglo_db:.4f}")

#SAMPLE COMPARISON

sample_size = min(5000, len(product_features))

sample_index = np.random.choice(
    len(product_features),
    size=sample_size,
    replace=False
)

print("KMeans unique clusters:", np.unique(kmeans_labels))
print("MiniBatchKMeans unique clusters:", np.unique(mbkmeans_labels))
print('agglomerative  unique clusters:',np.unique(agglo_labels))
print('db scan unique clusters:',np.unique(dbscan_labels))
comparison = pd.DataFrame({
    "Model": ["K-Means", "MiniBatchKMeans", "DBSCAN",'agglo'],
    "Silhouette (higher=better)": [kmeans_silhouette, mbkmeans_silhouette,dbscan_silhouette,agglo_silhouette],
    "Davies-Bouldin (lower=better)": [kmeans_db, mbkmeans_db, dbscan_db,agglo_db],
    "N clusters": [N_CLUSTERS, N_CLUSTERS, n_dbscan_clusters,N_CLUSTERS],
    "Unassigned users": [0, 0, n_noise,0],
    "Scales to full 20k+ users?": ["Yes", "Yes", "Yes",'Partially'],
})
comparison

score_df = pd.DataFrame({
    "Model": [
        "KMeans",
        "MiniBatchKMeans",
        "DBSCAN",
        'agglomerative'
    ],
    "Silhouette Score": [
        round(kmeans_silhouette, 4),
        round(mbkmeans_silhouette, 4),
        round(dbscan_silhouette, 4) if "dbscan_silhouette" in globals() else "Not valid",
        round(agglo_silhouette,4)
    ],
    "Davies-Bouldin Score": [
        round(kmeans_db, 4),
        round(mbkmeans_db, 4),
        round(dbscan_db, 4) if "dbscan_db" in globals() else "Not valid",
        round(agglo_db,4)
    ]
})

score_df

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    data=score_df,
    x="Model",
    y="Silhouette Score",
    palette="Blues"
)

plt.title("Silhouette Score Comparison")
plt.xlabel("Clustering Model")
plt.ylabel("Silhouette Score")

for container in ax.containers:
    ax.bar_label(container, fmt="%.4f", padding=3)

plt.show()

plt.figure(figsize=(7, 5))

ax = sns.barplot(
    data=score_df,
    x="Model",
    y="Davies-Bouldin Score",
    palette="Oranges"
)

plt.title("Davies-Bouldin Score Comparison")
plt.xlabel("Clustering Model")
plt.ylabel("Davies-Bouldin Score")

for container in ax.containers:
    ax.bar_label(container, fmt="%.4f", padding=3)

plt.show()

"""I evaluated clustering using two metrics. Silhouette Score should be higher. Davies-Bouldin Score should be lower. These metrics help compare which clustering model forms better product groups."""

df_filtered = df_filtered.copy()
df_filtered['cluster'] = df_filtered['userid'].map(dict(zip(user_ids, kmeans_labels)))

TOP_N_PER_CLUSTER = 50

def build_popularity_tables(data, cluster_col='cluster'):

    cluster_stats = (data.groupby([cluster_col, 'productid'])['ratings']
                          .agg(['mean', 'count']).reset_index())
    cluster_stats['score'] = cluster_stats['mean'] * np.log1p(cluster_stats['count'])

    cluster_top = {
        c: grp.sort_values('score', ascending=False).head(TOP_N_PER_CLUSTER)['productid'].tolist()
        for c, grp in cluster_stats.groupby(cluster_col)
    }

    global_stats = data.groupby('productid')['ratings'].agg(['mean', 'count']).reset_index()
    global_stats['score'] = global_stats['mean'] * np.log1p(global_stats['count'])
    global_top = global_stats.sort_values('score', ascending=False).head(TOP_N_PER_CLUSTER)['productid'].tolist()

    return cluster_top, global_top

cluster_top_products, global_top_products = build_popularity_tables(df_filtered)
print(f"Built per-cluster popularity lists for {len(cluster_top_products)} clusters")
print(f"Global fallback list: {len(global_top_products)} products")

"""This table shows which cluster each product belongs to. Products in the same cluster have similar rating patterns."""

seen_by_user = df_filtered.groupby('userid')['productid'].apply(set).to_dict()
user_cluster_map = dict(zip(user_ids, kmeans_labels))

def recommend(user_id, k=10):
    """Top-k product recommendations for a user, filtering out products they've already rated."""
    already_seen = seen_by_user.get(user_id, set())

    if user_id in user_cluster_map:
        cluster = user_cluster_map[user_id]
        candidates = cluster_top_products.get(cluster, [])
    else:
        candidates = []  # user fell below the min-ratings filter entirely

    recs = [p for p in candidates if p not in already_seen][:k]

    # cold-start / not-enough-candidates fallback: top up with global popularity
    if len(recs) < k:
        backfill = [p for p in global_top_products if p not in recs and p not in already_seen]
        recs += backfill[:k - len(recs)]

    return recs

# Demo on a real user from the filtered dataset
demo_user = user_ids[0]
print(f"User: {demo_user}  |  cluster: {user_cluster_map[demo_user]}")
print(f"Already rated {len(seen_by_user[demo_user])} products")
print(f"Top 10 recommendations: {recommend(demo_user, k=10)}")

"""This function recommends products from the same cluster. The idea is that products in the same cluster are similar, so they can be recommended together."""

comparison = pd.DataFrame({
    "Model": [
        "Popularity-Based Recommendation",
        "Item-Item Collaborative Filtering",
        "KMeans Clustering",
        "MiniBatchKMeans Clustering",
        "Agglomerative Clustering"
    ],

    "Purpose": [
        "Recommend generally popular products",
        "Recommend products similar to a selected product",
        "Group similar products into clusters",
        "Group similar products faster for large data",
        "Group similar users or products step by step by merging the closest items into clusters"
    ],

    "Personalized": [
        "No",
        "Yes",
        "Partially",
        "Partially",
        "Partially"
    ],

    "Evaluation": [
        "Average rating and rating count",
        "Cosine similarity",
        "Silhouette and Davies-Bouldin score",
        "Silhouette and Davies-Bouldin score",
        "Silhouette and Davies-Bouldin score"
    ],

    "Advantage": [
        "Simple and fast",
        "Gives similarity-based recommendations",
        "Easy to understand clusters",
        "Better for large datasets",
        "Shows hierarchical relationship between clusters"
    ],

    "Limitation": [
        "Same recommendations for everyone",
        "Needs enough rating history",
        "Slower on large data",
        "Approximate clustering",
        "Very slow and memory-heavy for large datasets"
    ]
})

comparison

comparison

evaluation_df = pd.DataFrame({
    "Model": ["KMeans", "MiniBatchKMeans",'dbscan','agglomerative'],
    "Silhouette Score": [kmeans_silhouette, mbkmeans_silhouette,dbscan_silhouette,agglo_silhouette],
    "Davies-Bouldin Score": [kmeans_db, mbkmeans_db,dbscan_db,agglo_db],
})

evaluation_df

popularity_eval = {
    "Model": "Popularity-Based Recommendation",
    "Evaluation Metric": "Average rating and rating count",
    "Average Rating of Top Products": round(popular_products["average_rating"].head(10).mean(), 4),
    "Average Rating Count of Top Products": round(popular_products["rating_count"].head(10).mean(), 2)
}

popularity_eval

item_based_eval = {
    "Model": "Item-Item Collaborative Filtering",
    "Evaluation Metric": "Cosine similarity",
    "Average Similarity Score": round(similar_products["similarity_score"].mean(), 4),
    "Highest Similarity Score": round(similar_products["similarity_score"].max(), 4)
}

item_based_eval

recommendation_eval_df = pd.DataFrame([
    popularity_eval,
    item_based_eval
])

recommendation_eval_df

import os
import pickle
import numpy as np

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

mbkmeans_user_cluster_map = dict(zip(user_ids, mbkmeans_labels))

df_model_save = df_filtered.copy()
df_model_save["mbkmeans_cluster"] = df_model_save["userid"].map(mbkmeans_user_cluster_map)


def build_cluster_top_products(data, cluster_col, top_n=50):
    cluster_stats = (
        data.groupby([cluster_col, "productid"])["ratings"]
        .agg(["mean", "count"])
        .reset_index()
    )

    cluster_stats["score"] = cluster_stats["mean"] * np.log1p(cluster_stats["count"])

    cluster_top_products = {
        cluster: group.sort_values("score", ascending=False)
        .head(top_n)["productid"]
        .tolist()
        for cluster, group in cluster_stats.groupby(cluster_col)
    }

    return cluster_top_products


mbkmeans_cluster_top_products = build_cluster_top_products(
    df_model_save,
    "mbkmeans_cluster"
)

model_bundle = {
    "kmeans_model": kmeans_model,
    "minibatch_kmeans_model": mbkmeans_model,
    "svd_model": svd_model,
    "cosine_model": item_model,

    "user_item_matrix": user_item_matrix,

    "user_ids": user_ids,
    "product_ids": product_ids,
    "user_to_idx": user_to_idx,
    "product_to_idx": product_to_idx,

    "kmeans_labels": kmeans_labels,
    "minibatch_labels": mbkmeans_labels,

    "kmeans_user_cluster_map": user_cluster_map,
    "minibatch_user_cluster_map": mbkmeans_user_cluster_map,

    "kmeans_cluster_top_products": cluster_top_products,
    "minibatch_cluster_top_products": mbkmeans_cluster_top_products,

    "global_top_products": global_top_products,
    "seen_by_user": seen_by_user,
    "popular_products": popular_products
}

pickle_path = os.path.join(MODEL_DIR, "product_recommendation_models.pkl")

with open(pickle_path, "wb") as file:
    pickle.dump(model_bundle, file)

print("Model pickle file saved successfully:")
print(pickle_path)

with open("models/product_recommendation_models.pkl", "rb") as file:
    loaded_model_bundle = pickle.load(file)

print("Pickle loaded successfully")
print("Available objects:")
print(loaded_model_bundle.keys())

import shutil
import joblib
from google.colab import files

shutil.make_archive("models", "zip", "models")
joblib.dump(model_bundle,'models/product_recommendation_models.joblib')
files.download('models/product_recommendation_models.joblib')







"""## model deployment"""

