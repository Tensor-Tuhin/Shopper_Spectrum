# Importing the necessary libraries
import numpy as np
import joblib
import streamlit as st
from utils.recommender import recommended_products


# Loading the pre-trained model and other saved artifacts for Customer Segmentation
@st.cache_resource
def load_segmentation_artifacts():
    kmeans = joblib.load('../data/artifacts_model/kmeans_model.pkl')                        # KMeans model
    cluster_labels = joblib.load('../data/artifacts_model/cluster_label_mapping.pkl')       # Cluster label mapping for the segments
    scaler = joblib.load('../data/artifacts_model/rfm_scaler.pkl')                          # Scaler for RFM features
    return kmeans, cluster_labels, scaler

kmeans, cluster_labels, scaler = load_segmentation_artifacts()


# Loading the artifacts for Product Recommendation
@st.cache_resource
def load_recommendation_artifacts():
    product_similarity = joblib.load('../data/artifacts_model/product_similarity.pkl')      # Product similarity dataframe/matrix
    product_index = joblib.load('../data/artifacts_model/product_list.pkl')                 # List of products
    return product_similarity, product_index

product_similarity, product_index = load_recommendation_artifacts()


# App Layout
tab1, tab2 = st.tabs(["Customer Segmentation", "Product Recommendation"])


# tab1: Customer Segmentation
with tab1:
    st.header('👦🏽Customer Segmentation👱🏽')

    recency = st.number_input('Recency (in days):', min_value=0)
    frequency = st.number_input('Frequency (number of purchases):', min_value=0)
    monetary = st.number_input('Monetary Value (total spend):', min_value=0.0)

    if st.button('Predict Segment'):
        # Preparing the input data
        x = np.array([[recency, frequency, monetary]])
        x_scaled = scaler.transform(x)

        # Predicting the clusters
        cluster = kmeans.predict(x_scaled)[0]
        segment = cluster_labels[cluster]
        st.success(f'The customer belongs to **{segment}** segment')


# tab2: Product Recommendation
with tab2:
    st.header('🛍️Product Recommendation🛒')

    product_name = st.selectbox('Select a Product:', product_index)

    if st.button('Get Recommendations'):
        recommendations = recommended_products(product_name, product_similarity)
        if isinstance(recommendations, str):                # Checks if an error message is returned
            st.error(recommendations)
        else:
            st.success('Top 5 similar products:')
            for i, prod in enumerate(recommendations, start=1):
                st.write(f"{i}. {prod}")