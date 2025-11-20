# nba_bin_pca.py

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns

df = pd.read_csv("nba_bin_df.csv")

numeric = ["LOC_X", "LOC_Y", "SHOT_DISTANCE"]
X = df[numeric]
y = df["SHOT_MADE_FLAG"]

X_scaled = StandardScaler().fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PC1": X_pca[:,0],
    "PC2": X_pca[:,1],
    "class": y
})

sns.scatterplot(
    data=pca_df.sample(min(5000, len(pca_df))),
    x="PC1", y="PC2", hue="class",
    palette={0:"red", 1:"green"}, alpha=0.4, s=10
)
plt.title("PCA (Numeric Features Only)")
plt.savefig("pca_plot.png", dpi=200)
plt.close()

pca_df.to_csv("pca_df.csv", index=False)
