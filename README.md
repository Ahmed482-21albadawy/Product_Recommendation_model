# 🛍️ Product Recommendation System

This project is a personalized product recommendation system built using **Collaborative Filtering** and **Truncated SVD**. It analyzes historical user-product ratings to predict future preferences and recommend the most relevant products to each user.

🧠 The system uses **matrix factorization** to extract hidden patterns between users and products. By reducing dimensionality, it provides smart recommendations even with sparse data.

🔧 **Key Features:**
- Efficient product recommendations based on past user behavior
- Handles missing ratings using user-based averages
- Evaluates performance using MAE, RMSE, R², and classification accuracy
- New users get random product suggestions as a fallback
- Integration with a SQLite database to display product names

📁 The trained model and user-product matrix are saved for reuse, making the system fast and production-ready.

This system is ideal for small-to-medium e-commerce platforms looking to deliver personalized user experiences with minimal setup.

---
