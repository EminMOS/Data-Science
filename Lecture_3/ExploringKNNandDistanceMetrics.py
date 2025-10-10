import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pandas as pd


# Part 1: Euclidean Distance on Iris Dataset
# Load dataset
iris = load_iris()
X, y = iris.data[:, :2], iris.target   # use only 2 features for visualization

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train with Euclidean distance
knn_euclidean = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_euclidean.fit(X_train, y_train)
y_pred = knn_euclidean.predict(X_test)
acc_euclidean = accuracy_score(y_test, y_pred)

print("Part 1 - Euclidean on Iris accuracy:", acc_euclidean)

# Part 2: Manhattan Distance on Grid-like Dataset

# Synthetic dataset
X_syn, y_syn = make_classification(n_samples=300, n_features=2, n_classes=2, n_redundant=0, n_informative=2,
                                   n_clusters_per_class=1, random_state=42)
X_syn = np.round(X_syn)   # grid-like

X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(X_syn, y_syn, test_size=0.2, random_state=42)

knn_manhattan = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
knn_manhattan.fit(X_train_syn, y_train_syn)
y_pred_syn = knn_manhattan.predict(X_test_syn)
acc_manhattan = accuracy_score(y_test_syn, y_pred_syn)

print("Part 2 - Manhattan on Grid accuracy:", acc_manhattan)

# Part 3: Decision Boundary Visualization
def plot_decision_boundary(clf, X, y, title, class_names=None):
    h = 0.1
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Set1)

    # Add legend
    if class_names is not None:
        handles, labels = scatter.legend_elements()
        plt.legend(handles, class_names, title="Classes")

    plt.title(title)
    plt.show()

# Visualize Iris (Euclidean)
plot_decision_boundary(knn_euclidean, X, y, "Iris - Euclidean", class_names=iris.target_names)

# Visualize Grid dataset (Manhattan)
plot_decision_boundary(knn_manhattan, X_syn, y_syn, "Grid - Manhattan", class_names=["Class 0", "Class 1"])

# Compare with Euclidean on grid
knn_euclidean_grid = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_euclidean_grid.fit(X_train_syn, y_train_syn)
plot_decision_boundary(knn_euclidean_grid, X_syn, y_syn, "Grid - Euclidean", class_names=["Class 0", "Class 1"])

# Part 4: Experimenting with K

k_values = [1, 3, 5, 7, 15]
results = []

for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    # knn_k.fit(X_train, y_train)
    # y_pred_k = knn_k.predict(X_test)
    # acc = accuracy_score(y_test, y_pred_k)
    scores = cross_val_score(knn_k, X, y, cv=5)
    results.append([k, scores.mean()])

acc_table = pd.DataFrame(results, columns=["k", "Accuracy (Iris, Euclidean)"])
print("\nAccuracy for different k values:")
print(acc_table)