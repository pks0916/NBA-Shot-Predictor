import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

# Load base dataframe
df = pd.read_csv("nba_bin_df.csv")

# Only numeric spatial features for PCA
numeric = ["LOC_X", "LOC_Y", "SHOT_DISTANCE"]
X = df[numeric]
y = df["SHOT_MADE_FLAG"]

# Standardize before PCA
X_scaled = StandardScaler().fit_transform(X)

# 2-D PCA so we can plot
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained variance ratio (PC1, PC2):", pca.explained_variance_ratio_)

# Build PCA dataframe
pca_df = pd.DataFrame(
    {
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "class": y,
    }
)

# Save for classifier script
pca_df.to_csv("pca_df.csv", index=False)
print("Saved pca_df.csv with shape:", pca_df.shape)

# Plot PCA scatter (sample for readability)
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=pca_df.sample(min(5000, len(pca_df))),
    x="PC1",
    y="PC2",
    hue="class",
    palette={0: "red", 1: "green"},
    s=10,
    alpha=0.4,
)
plt.title("PCA of Shot Location Features")
plt.tight_layout()
plt.savefig("pca_plot.png", dpi=200)
plt.close()
