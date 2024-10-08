import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.01, random_state=10)

# Train model
clf = DecisionTreeClassifier().fit(X_train, Y_train)

# Print accuracy
print(f'Accuracy: {clf.score(X_test, Y_test):.2f}')

# Plot tree
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.show()