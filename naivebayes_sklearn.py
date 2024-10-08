import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

X = np.array([
    [0, 1.6], [1, 2], [0, 1.9], [0, 1.88], [0, 1.7],
    [1, 1.85], [0, 1.6], [1, 1.7], [1, 2.2], [1, 2.1],
    [0, 1.8], [1, 1.95], [0, 1.9], [0, 1.8], [0, 1.75]
])

Y = np.array([
    "S", "T", "M", "M", "S", "M", "S", "S",
    "T", "T", "M", "M", "M", "M", "M"
])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

clf = GaussianNB()
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
my_prediction = clf.predict(np.array([[1, 1.95]]))

accuracy = accuracy_score(Y_test, Y_pred)
print("My prediction:", my_prediction)
print("Accuracy:", accuracy)