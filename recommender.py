def recommended_products(product_name, product_similarity, n=5):
    """
    This function takes a product name and returns the top 5 similar products
    based on cosine similarity.
    """
    product_name = product_name.strip().lower()

    if product_name not in product_similarity.index:
        return "Product not found in the database."
    
    # Get the similarity scores for the given product
    similarity_scores = product_similarity[product_name]
    
    # Sort the products based on similarity scores
    similar_products = similarity_scores.sort_values(ascending=False)
    
    # Exclude the input product itself and get the top n similar products
    top_similar_products = similar_products.iloc[1:n+1]
    
    return list(top_similar_products.index)