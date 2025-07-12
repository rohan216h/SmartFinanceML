# stock_recommender.py

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Mock User-Stock Interaction Data
data = {
    "User": ["U1", "U2", "U3", "U4", "U5"],
    "AAPL": [1, 0, 1, 0, 1],
    "TSLA": [1, 1, 0, 1, 0],
    "MSFT": [0, 1, 1, 0, 1],
    "AMZN": [1, 0, 0, 1, 0],
    "NFLX": [0, 0, 1, 1, 1]
}

df = pd.DataFrame(data).set_index("User")
print("🔍 User-Stock Interaction Matrix:")
print(df)

# -----------------------------
# 2. Compute Cosine Similarity Between Stocks
similarity_matrix = pd.DataFrame(
    cosine_similarity(df.T),
    index=df.columns,
    columns=df.columns
)

# inside your script after computing similarity_matrix:
plt.figure(figsize=(8, 6))
sns.heatmap(similarity_matrix, annot=True, cmap="coolwarm")
plt.title("Stock Similarity (Cosine)")
plt.tight_layout()
plt.savefig("reports/stock_similarity_matrix.png")
plt.close()


# -----------------------------
# 3. Recommend Similar Stocks
def recommend_similar(stock, top_n=2):
    if stock not in similarity_matrix.columns:
        return f"❌ Stock '{stock}' not in dataset."
    
    sims = similarity_matrix[stock].drop(stock)
    top_stocks = sims.sort_values(ascending=False).head(top_n)
    return top_stocks

# -----------------------------
# Try a Recommendation
target_stock = "AAPL"
print(f"\n📊 Because you interacted with '{target_stock}', you might also like:")
print(recommend_similar(target_stock))
