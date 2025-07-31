import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# Load Data
products = pd.read_csv("Products.csv")
user_purchases = pd.read_csv("User_Purchases.csv")

# Convert 'Timestamp' to datetime format if exists
if "Timestamp" in user_purchases.columns:
    user_purchases["Timestamp"] = pd.to_datetime(user_purchases["Timestamp"])

# Sort by User ID, Product ID, and Timestamp (latest first)
user_purchases = user_purchases.sort_values(by=["UserID", "ProductID", "Timestamp"], ascending=[True, True, False])

# Drop duplicates, keeping only the latest rating per user-product
user_purchases = user_purchases.drop_duplicates(subset=["UserID", "ProductID"], keep="first")


# Create User-Product Matrix(with zero values)
user_product_matrix = user_purchases.pivot(index="UserID", columns="ProductID", values="Rating").fillna(0)

# Convert to numpy array
matrix = user_product_matrix.values

# save the matrix file
np.save("user_product_matrix.npy", matrix)
# save the columns & rows structure
np.save("user_product_matrix_columns.npy", user_product_matrix.columns)  # Save column order
np.save("user_product_matrix_index.npy", user_product_matrix.index)  # Save row order

# save the matrix in the same variable( with mean values rather than zero values)
user_product_matrix = user_purchases.pivot(index="UserID", columns="ProductID", values="Rating")
user_product_matrix = user_product_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

matrix = user_product_matrix.values

# Train-Test Split (Ensuring No Data Leakage)
train_indices, test_indices = train_test_split(range(matrix.shape[0]), test_size=0.3, random_state=42)
train_data = matrix[train_indices]
test_data = matrix[test_indices]

# Train SVD Model
n_components = 40
svd = TruncatedSVD(n_components=n_components, random_state=42)
train_transformed = svd.fit_transform(train_data)
reconstructed_train = np.dot(train_transformed, svd.components_)

# Transform Test Data
test_transformed = svd.transform(test_data)
reconstructed_test = np.dot(test_transformed, svd.components_)


# train_rmse = mean_squared_error(train_data, reconstructed_train, squared=False)
# test_rmse =  mean_squared_error(test_data, reconstructed_test, squared=False)


# print(f"Train RMSE: {train_rmse:.2f}")
# print(f"Test RMSE: {test_rmse:.2f}")

# Calculate the metrics for the train data
train_mae = mean_absolute_error(train_data, reconstructed_train)  
train_mse = mean_squared_error(train_data, reconstructed_train)
train_rmse = mean_squared_error(train_data, reconstructed_train, squared=False)  # RMSE
train_r2 = r2_score(train_data, reconstructed_train)  # R² score

print("Train Data Metrics:")
print(f"Mean Absolute Error (MAE): {train_mae:.2f}")
print(f"Mean Squared Error (MSE): {train_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {train_rmse:.2f}")
print(f"R² Score: {train_r2:.2f}")

# Calculate the metrics for the test data
test_mae = mean_absolute_error(test_data, reconstructed_test)  # Use actual test data and predicted values (reconstructed)
test_mse = mean_squared_error(test_data, reconstructed_test)
test_rmse = mean_squared_error(test_data, reconstructed_test, squared=False)  # RMSE
test_r2 = r2_score(test_data, reconstructed_test)  # R² score

print("Test Data Metrics:")
print(f"Mean Absolute Error (MAE): {test_mae:.2f}")
print(f"Mean Squared Error (MSE): {test_mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {test_rmse:.2f}")
print(f"R² Score: {test_r2:.2f}")

# Calculating the Accuracy #

from sklearn.metrics import accuracy_score

# Convert ratings into binary: liked (1) or not liked (0)
threshold = 3.0
y_test_binary = (test_data >= threshold).astype(int)
y_pred_binary = (reconstructed_test >= threshold).astype(int)

accuracy_test = accuracy_score(y_test_binary.flatten(), y_pred_binary.flatten())
print(f"Test_Accuracy: {accuracy_test*100:.2f}%")


y_train_binary = (train_data >= threshold).astype(int)
y_pred_binary2 = (reconstructed_train >= threshold).astype(int)

accuracy_train = accuracy_score(y_train_binary.flatten(), y_pred_binary2.flatten())
print(f"Train_Accuracy: {accuracy_train*100:.2f}%")


import joblib

# Save the trained SVD model
joblib.dump(svd, "svd_model.pkl")

