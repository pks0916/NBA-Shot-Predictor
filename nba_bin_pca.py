# nba_bin_pca.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

df = pd.read_csv("nba_shots.csv")
df = df[df["SHOT_ATTEMPTED_FLAG"] == 1].copy()
df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)

numeric_features = ["LOC_X", "LOC_Y", "SHOT_DISTANCE"]

X = df[numeric_features]
y = df["SHOT_MADE_FLAG"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance:", pca.explained_variance_ratio_)

df_pca = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "class": y
})

# scatter
plt.figure(figsize=(7,6))
sns.scatterplot(data=df_pca, x="PC1", y="PC2", hue="class", palette={0:"red", 1:"green"}, alpha=0.3)
plt.title("PCA of NBA Shots (Numeric Features Only)")
plt.show()

def get_pca_df():
    return df_pca
