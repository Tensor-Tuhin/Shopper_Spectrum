# Shopper_Spectrum
SHOPPER SPECTRUM: CUSTOMER SEGMENTATION AND PRODUCT RECOMMENDATIONS IN E-COMMERCE.

This project analyzes online retail transaction data to understand customer purchasing behavior, segment customers using RFM (Recency, Frequency, Monetary) analysis, and recommend similar products using item-based collaborative filtering. The final solution is deployed as an interactive Streamlit application.


Objectives:

- Segment customers based on purchasing behavior using unsupervised machine learning algorithm (KMeans)
- Identify high-value, regular, occasional, and at-risk customers
- Recommend relevant products based on historical purchase patterns


Methodology:

- Customer Segmentation -
    - Feature engineering using RFM metrics
    - Data standardization using StandardScaler
    - KMeans clustering with optimal cluster selection using:
        - Elbow Method
        - Silhouette Score
- Recommendation System -
    - Item-based collaborative filtering
    - Product–Customer matrix built from transaction quantities
    - Cosine similarity used to recommend top 5 similar products
 

Streamlit Application:

- Product Recommendation: Select a product to receive 5 similar product suggestions
- Customer Segmentation: Input RFM values to predict the customer segment


Tools:

- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Streamlit
- Joblib


Deploying the app:

- Make sure to have the mentioned tools installed
- In terminal -> streamlit run app.py
