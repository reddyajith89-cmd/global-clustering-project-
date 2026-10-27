"""
Global Clustering Project - training script (PCA + KMeans + DBSCAN)

Run:  python train_model.py

Input : World_development_mesurement.xlsx  (put it in this folder)
Output: country_results.csv, kmeans_model.pkl, dbscan_model.pkl,
        scaler.pkl, pca.pkl, and graphs inside the plots/ folder
"""

import os
import pickle
import urllib.request

import matplotlib
matplotlib.use("Agg")  # save graphs to files instead of opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

DATA_FILE = "World_development_mesurement.xlsx"

# GitHub "raw" link of the Excel file (change this to your own link)
DATA_URL = "https://raw.githubusercontent.com/ajithkumar564/global-clustering-project/main/World_development_mesurement.xlsx"

os.makedirs("plots", exist_ok=True)


def save_plot(name):
    plt.savefig(os.path.join("plots", name), dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved graph: plots/" + name)


# ---------------------------------------------------
# Get the data file
# ---------------------------------------------------
if os.path.exists(DATA_FILE):
    print("Data file is already here.")
else:
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
        print("Data file downloaded.")
    except Exception:
        raise SystemExit(
            f"Could not download the data. Please put '{DATA_FILE}' in this folder and run again."
        )

# ---------------------------------------------------
# Step 2: Load the data
# ---------------------------------------------------
data = pd.read_excel(DATA_FILE)
print("Data shape:", data.shape)

# ---------------------------------------------------
# Step 3: Keep Country and Year separate
# The Excel file has no Year column, but rows are in year order (2000, 2001, ...)
# ---------------------------------------------------
country = data["Country"]
year = 2000 + data.groupby("Country").cumcount()
data = data.drop("Country", axis=1)

print("Rows per year:")
print(year.value_counts().sort_index())

# ---------------------------------------------------
# Step 4: Clean the data (remove $ , % and change text to numbers)
# ---------------------------------------------------
for col in data.columns:
    if not pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].astype(str).str.replace("%", "", regex=False)
        data[col] = data[col].str.replace("$", "", regex=False)
        data[col] = data[col].str.replace(",", "", regex=False)
        data[col] = pd.to_numeric(data[col], errors="coerce")

# ---------------------------------------------------
# Step 5: Fill missing values
# ---------------------------------------------------
# Save the real GDP before filling (the app should not show made-up GDP)
gdp_original = data["GDP"].copy()

data = data.replace([np.inf, -np.inf], np.nan)
data = data.fillna(data.median())
print("Missing values left:", int(data.isnull().sum().sum()))

# ---------------------------------------------------
# Step 6: Correlation heatmap
# ---------------------------------------------------
plt.figure(figsize=(15, 10))
sns.heatmap(data.corr(numeric_only=True), cmap="coolwarm")
save_plot("correlation_heatmap.png")

# ---------------------------------------------------
# Step 7: Scale the data
# ---------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# ---------------------------------------------------
# Step 8: PCA (keep 90% of the information)
# ---------------------------------------------------
pca = PCA(n_components=0.9)
x_pca = pca.fit_transform(scaled_data)

print("Original columns:", scaled_data.shape[1])
print("Columns after PCA:", x_pca.shape[1])

explained_variance = pca.explained_variance_ratio_
print("Total variance covered:", np.sum(explained_variance))

plt.plot(np.cumsum(explained_variance))
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("Variance explained by PCA")
save_plot("pca_variance.png")

plt.scatter(x_pca[:, 0], x_pca[:, 1])
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.title("PCA projection in 2D")
save_plot("pca_2d.png")

# ---------------------------------------------------
# Step 9: KMeans clustering
# ---------------------------------------------------
wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, n_init=10, random_state=42)
    model.fit(x_pca)
    wcss.append(model.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
save_plot("kmeans_elbow.png")

# Final KMeans model with 3 clusters
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans_labels = kmeans.fit_predict(x_pca)
data["Cluster_KMeans"] = kmeans_labels
print("KMeans cluster sizes:")
print(data["Cluster_KMeans"].value_counts())

plt.scatter(x_pca[:, 0], x_pca[:, 1], c=kmeans_labels)
plt.title("KMeans clusters after PCA")
save_plot("kmeans_clusters.png")

# ---------------------------------------------------
# Step 10: DBSCAN clustering
# ---------------------------------------------------
neighbors = NearestNeighbors(n_neighbors=10)
neighbors.fit(x_pca)
distances, indices = neighbors.kneighbors(x_pca)

distances = np.sort(distances[:, 9])
plt.plot(distances)
plt.xlabel("Points")
plt.ylabel("Distance")
plt.title("K-distance Graph")
save_plot("dbscan_k_distance.png")

# Tuning: try many eps and min_samples values, keep the best silhouette score
eps_values = [1.0, 1.2, 1.3, 1.4, 1.5, 1.7, 1.9, 2.0]
min_sample_range = range(1, 20)
output = []

for eps_val in eps_values:
    for min_samp in min_sample_range:
        labels = DBSCAN(eps=eps_val, min_samples=min_samp).fit(x_pca).labels_

        # Silhouette score needs at least 2 groups, so skip the bad combinations
        if len(set(labels)) < 2 or len(set(labels)) > len(labels) - 1:
            continue

        score = silhouette_score(x_pca, labels)
        output.append([eps_val, min_samp, score])

df_results = pd.DataFrame(output, columns=["eps", "min_samples", "score"])
if df_results.empty:
    raise SystemExit("DBSCAN tuning found no valid eps/min_samples combination.")

best = df_results.sort_values("score", ascending=False).iloc[0]
best_eps = best["eps"]
best_min_samples = int(best["min_samples"])
best_score = best["score"]
print(f"Best DBSCAN parameters: eps={best_eps}, min_samples={best_min_samples}, score={best_score:.4f}")

plt.figure(figsize=(14, 8))
scatter = plt.scatter(df_results["eps"], df_results["min_samples"],
                      c=df_results["score"], cmap="RdYlGn",
                      s=200, alpha=0.7, edgecolors="black", linewidth=0.5)
plt.scatter([best_eps], [best_min_samples], color="red", s=500, marker="*",
            edgecolors="black", linewidth=2,
            label=f"Best: eps={best_eps}, min_samples={best_min_samples}", zorder=5)
plt.xlabel("eps")
plt.ylabel("min_samples")
plt.title("DBSCAN Parameter Tuning")
plt.colorbar(scatter, label="Silhouette Score")
plt.legend()
plt.grid(True, alpha=0.3)
save_plot("dbscan_tuning.png")

# Final DBSCAN model with the best parameters
dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
dbscan_labels = dbscan.fit_predict(x_pca)
data["Cluster_DBScan"] = dbscan_labels
print("DBSCAN cluster sizes (-1 means noise):")
print(data["Cluster_DBScan"].value_counts())

# ---------------------------------------------------
# Step 11: Compare KMeans and DBSCAN
# ---------------------------------------------------
kmeans_score = silhouette_score(x_pca, kmeans_labels)
dbscan_score = silhouette_score(x_pca, dbscan_labels) if len(set(dbscan_labels)) > 1 else -1
print("KMeans Silhouette Score:", kmeans_score)
print("DBSCAN Silhouette Score:", dbscan_score)

# ---------------------------------------------------
# Step 12: Save results (used by app.py) and models
# ---------------------------------------------------
results = pd.DataFrame({
    "Country": country,
    "Year": year,
    "GDP": gdp_original,
    "Cluster_KMeans": kmeans_labels,
    "Cluster_DBScan": dbscan_labels,
})
results.to_csv("country_results.csv", index=False)
print("Saved country_results.csv")

for file_name, obj in [("kmeans_model.pkl", kmeans), ("dbscan_model.pkl", dbscan),
                       ("scaler.pkl", scaler), ("pca.pkl", pca)]:
    with open(file_name, "wb") as f:
        pickle.dump(obj, f)
print("Saved model files (.pkl)")

print("\nDone! Now run:  streamlit run app.py")
