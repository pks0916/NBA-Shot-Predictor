# nba_bin_classifiers_pca.py
from nba_bin_pca import get_pca_df
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics.pairwise import rbf_kernel

df = get_pca_df()

X = df[["PC1", "PC2"]]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Models
lr = LogisticRegression(max_iter=1000)
knn = KNeighborsClassifier(n_neighbors=7)
nb = GaussianNB()

lr.fit(X_train, y_train)
knn.fit(X_train, y_train)
nb.fit(X_train, y_train)

# Kernel LR
K_train = rbf_kernel(X_train, X_train, gamma=0.01)
K_test = rbf_kernel(X_test, X_train, gamma=0.01)
klr = LogisticRegression(C=10000, max_iter=2000)
klr.fit(K_train, y_train)

def eval(name, model, X_eval, kernel=False):
    if kernel:
        y_pred = model.predict(X_eval)
    else:
        y_pred = model.predict(X_eval)
    print("\n=== {} ===".format(name))
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

eval("LR + PCA", lr, X_test)
eval("KNN + PCA", knn, X_test)
eval("NB + PCA", nb, X_test)
eval("Kernel LR + PCA", klr, K_test, kernel=True)
