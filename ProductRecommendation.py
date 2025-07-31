import sqlite3
import numpy as np
import joblib
import random

# Load the trained SVD model
svd = joblib.load("svd_model.pkl")

# Load user and product mappings
product_ids = np.load("user_product_matrix_columns.npy", allow_pickle=True)
user_ids = np.load("user_product_matrix_index.npy", allow_pickle=True)
user_product_matrix = np.load("user_product_matrix.npy", allow_pickle=True)

# Connect to the SQLite database to get product names
def get_product_name(product_id):
    conn = sqlite3.connect("products.db")
    cursor = conn.cursor()
    cursor.execute("SELECT ProductName FROM products WHERE ProductID = ?", (product_id,))
    result = cursor.fetchone()
    # print(type(result))
    conn.close()
    return result[0] if result else "Unknown Product"

# Function to recommend top products for a user
def recommend_products_for_user(user_id, top_n=5):
    if user_id not in user_ids:
        # If user is new, return random products
        random_product_ids = random.sample(list(product_ids), top_n)
        # print(type(random_product_ids))
        return [get_product_name(int(pid)) for pid in random_product_ids]

    # Get the index of the user in the matrix
    user_index = np.where(user_ids == user_id)[0][0]
    # print(type(user_index))

    # Extract the user's existing ratings
    user_ratings = user_product_matrix[user_index]
    # print(type(user_ratings))

    # Transform and predict using SVD
    user_transformed = svd.transform(user_ratings.reshape(1, -1))
    # print(type(user_transformed))
    predicted_ratings = np.dot(user_transformed, svd.components_)
    # print(type(predicted_ratings))

    # Get top recommended product IDs (excluding already rated products)
    already_rated = user_ratings > 0  # Mask for products user has already rated
    predicted_ratings[0][already_rated] = -np.inf  # Ignore already rated products

    top_indices = np.argsort(predicted_ratings[0])[::-1][:top_n]
    recommended_product_ids = product_ids[top_indices]
    # print(type(recommended_product_ids))

    # Fetch product names from the database
    return [get_product_name(int(product_id)) for product_id in recommended_product_ids]

# Example: Get user input for user ID
user_id = int(input("Enter User ID: "))  # Taking user input
recommended = recommend_products_for_user(user_id)
# print(type(recommended))

print(f"\nTop Recommended Products for User {user_id}:")
for product in recommended:
    print(product)
