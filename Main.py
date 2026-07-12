# ==========================================================
# CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# ==========================================================

# ==========================
# Import Libraries
# ==========================

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings("ignore")

# ==========================
# Load Dataset
# ==========================

data = pd.read_csv("Mall_Customers.csv")

print("First 5 Rows")
print(data.head())

print("\nDataset Shape")
print(data.shape)

# ==========================
# Basic Information
# ==========================

print("\nInformation")
print(data.info())

print("\nMissing Values")
print(data.isnull().sum())

print("\nDuplicate Rows")
print(data.duplicated().sum())

print("\nStatistical Summary")
print(data.describe())

# ==========================
# Rename Columns (Optional)
# ==========================

data.columns = [
    "CustomerID",
    "Gender",
    "Age",
    "AnnualIncome",
    "SpendingScore"
]

# ==========================
# Exploratory Data Analysis
# ==========================

sns.set(style="whitegrid")

# Age Distribution
plt.figure(figsize=(6,4))
sns.histplot(data["Age"], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

# Annual Income Distribution
plt.figure(figsize=(6,4))
sns.histplot(data["AnnualIncome"], bins=20, kde=True)
plt.title("Annual Income Distribution")
plt.show()

# Spending Score Distribution
plt.figure(figsize=(6,4))
sns.histplot(data["SpendingScore"], bins=20, kde=True)
plt.title("Spending Score Distribution")
plt.show()

# Gender Count
plt.figure(figsize=(5,4))
sns.countplot(x="Gender", data=data)
plt.title("Gender Count")
plt.show()

# Pairplot
sns.pairplot(data.drop("CustomerID", axis=1), hue="Gender")
plt.show()

# Correlation Heatmap

numeric_data = data.select_dtypes(include=np.number)

plt.figure(figsize=(8,6))
sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==========================
# Preprocessing
# ==========================

encoder = LabelEncoder()

data["Gender"] = encoder.fit_transform(data["Gender"])

# Features used for clustering

X = data[["AnnualIncome", "SpendingScore"]]

# Standardization

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# ==========================
# Elbow Method
# ==========================

wcss = []

for i in range(1,11):
    model = KMeans(n_clusters=i, random_state=42)
    model.fit(X_scaled)
    wcss.append(model.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# ==========================
# KMeans Clustering
# ==========================

kmeans = KMeans(n_clusters=5, random_state=42)

clusters = kmeans.fit_predict(X_scaled)

data["Cluster"] = clusters

# ==========================
# Silhouette Score
# ==========================

score = silhouette_score(X_scaled, clusters)

print("\nSilhouette Score:", score)

# ==========================
# Cluster Summary
# ==========================

cluster_summary = data.groupby("Cluster").mean(numeric_only=True)

print("\nCluster Summary")

print(cluster_summary)

# ==========================
# Scatter Plot
# ==========================

plt.figure(figsize=(10,7))

sns.scatterplot(
    x="AnnualIncome",
    y="SpendingScore",
    hue="Cluster",
    palette="Set1",
    data=data,
    s=100
)

plt.scatter(
    scaler.inverse_transform(kmeans.cluster_centers_)[:,0],
    scaler.inverse_transform(kmeans.cluster_centers_)[:,1],
    s=300,
    c="black",
    marker="X",
    label="Centroids"
)

plt.title("Customer Segmentation")

plt.legend()

plt.show()

# ==========================
# Customers in each Cluster
# ==========================

print("\nCustomers in each Cluster")

print(data["Cluster"].value_counts())

# ==========================
# Save Result
# ==========================

data.to_csv("Customer_Segmentation_Result.csv", index=False)

print("\nResult saved successfully!")

print(data.head())

