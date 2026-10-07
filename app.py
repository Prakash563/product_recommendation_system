

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import joblib


st.set_page_config(
    page_title="Product Recommendation System",
    layout="wide"
)

st.title("Product Recommendation System")
st.write("This app uses saved trained models for product recommendation.")



@st.cache_resource
def load_model_bundle():
    model_bundle = joblib.load("models/product_recommendation_models.joblib")
    return model_bundle

import streamlit as st

@st.cache_resource
def load_model_bundle():
    return joblib.load("models/product_recommendation_models.joblib")

model_bundle = load_model_bundle()

kmeans_model = model_bundle["kmeans_model"]
minibatch_kmeans_model = model_bundle["minibatch_kmeans_model"]
svd_model = model_bundle["svd_model"]
cosine_model = model_bundle["cosine_model"]

user_item_matrix = model_bundle["user_item_matrix"]

user_ids = model_bundle["user_ids"]
product_ids = model_bundle["product_ids"]

user_to_idx = model_bundle["user_to_idx"]
product_to_idx = model_bundle["product_to_idx"]

kmeans_user_cluster_map = model_bundle["kmeans_user_cluster_map"]
minibatch_user_cluster_map = model_bundle["minibatch_user_cluster_map"]

kmeans_cluster_top_products = model_bundle["kmeans_cluster_top_products"]
minibatch_cluster_top_products = model_bundle["minibatch_cluster_top_products"]

global_top_products = model_bundle["global_top_products"]
popular_products = model_bundle["popular_products"]
seen_by_user = model_bundle["seen_by_user"]

def recommend_for_user(user_id, model_name="KMeans", top_n=10):
    already_seen = seen_by_user.get(user_id, set())

    if model_name == "KMeans":
        user_cluster_map = kmeans_user_cluster_map
        cluster_top_products = kmeans_cluster_top_products
    else:
        user_cluster_map = minibatch_user_cluster_map
        cluster_top_products = minibatch_cluster_top_products

    if user_id in user_cluster_map:
        cluster = user_cluster_map[user_id]
        candidate_products = cluster_top_products.get(cluster, [])
    else:
        cluster = "New / Unknown User"
        candidate_products = []

    recommendations = [
        product for product in candidate_products
        if product not in already_seen
    ][:top_n]

    if len(recommendations) < top_n:
        fallback_products = [
            product for product in global_top_products
            if product not in recommendations and product not in already_seen
        ]

        recommendations = recommendations + fallback_products[:top_n - len(recommendations)]

    return cluster, recommendations

def recommend_similar_products(product_id, top_n=10):
    if product_id not in product_to_idx:
        return pd.DataFrame(columns=["productid", "similarity_score"])

    product_index = product_to_idx[product_id]

    n_neighbors = min(top_n + 1, len(product_ids))

    distances, indices = cosine_model.kneighbors(
        user_item_matrix.T[product_index],
        n_neighbors=n_neighbors
    )

    recommended_indices = indices.flatten()[1:]
    similarity_scores = 1 - distances.flatten()[1:]

    result = pd.DataFrame({
        "productid": product_ids[recommended_indices],
        "similarity_score": similarity_scores
    })

    return result

def get_popular_products(top_n=10):
    return popular_products.head(top_n)

st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox(
    "Choose clustering model",
    ["KMeans", "MiniBatchKMeans"]
)

top_n = st.sidebar.slider(
    "Number of recommendations",
    min_value=5,
    max_value=20,
    value=10
)

tab1, tab2, tab3 = st.tabs([
    "User Recommendations",
    "Similar Products",
    "Popular Products"
])

with tab1:
    st.subheader("User-Based Product Recommendations")

    user_id_input = st.selectbox(
    "Select User ID",
    options=sorted(user_ids)
)

    if st.button("Recommend Products for User"):
        cluster, recommendations = recommend_for_user(
            user_id_input,
            model_name=model_choice,
            top_n=top_n
        )

        st.write("Selected Model:", model_choice)
        st.write("User Cluster:", cluster)

        recommendation_df = pd.DataFrame({
            "recommended_productid": recommendations
        })

        st.dataframe(recommendation_df, use_container_width=True)

with tab2:
    st.subheader("Cosine Similarity Product Recommendations")

   product_id_input = st.selectbox(
        "Select Product ID",
        options=sorted(product_ids)
    )


    if st.button("Find Similar Products"):
        similar_products = recommend_similar_products(
            product_id_input,
            top_n=top_n
        )

        if similar_products.empty:
            st.warning("Product ID not found.")
        else:
            st.dataframe(similar_products, use_container_width=True)

with tab3:
    st.subheader("Popularity-Based Recommendations")

    top_popular_products = get_popular_products(top_n)

    st.dataframe(top_popular_products, use_container_width=True)

