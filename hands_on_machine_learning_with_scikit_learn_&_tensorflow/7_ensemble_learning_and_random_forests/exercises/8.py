import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier


mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.target = mnist.target.astype(np.uint8)

X_train_val, X_test, y_train_val, y_test = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=10000, random_state=42)

random_forest_clf = RandomForestClassifier(n_estimators=100, random_state=42) # 随机森林
extra_trees_clf = ExtraTreesClassifier(n_estimators=100, random_state=42) # 极端随机树
svm_clf = LinearSVC(max_iter=100, tol=20, random_state=42) # SVM分类器

estimators = [random_forest_clf, extra_trees_clf, svm_clf]
for estimator in estimators:
    estimator.fit(X_train, y_train)
    print(estimator.__class__.__name__, estimator.score(X_val, y_val))

estimators = [
    ("random_forest_clf", random_forest_clf),
    ("extra_trees_clf", extra_trees_clf),
    ("svm_clf", svm_clf),
]
voting_clf = VotingClassifier(estimators)
voting_clf.fit(X_train, y_train)
print(voting_clf.score(X_val, y_val))
