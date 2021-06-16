from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

params = {
    'max_leaf_nodes': list(range(2, 100)), 
    'min_samples_split': [2, 3, 4]
}
tree_clf = DecisionTreeClassifier(random_state=42)
grid_search_cv = GridSearchCV(tree_clf, params, verbose=1, cv=3, n_jobs=-1)

grid_search_cv.fit(X_train, y_train)
grid_search_cv.best_estimator_

print(grid_search_cv.best_estimator_.score(X_test, y_test))
