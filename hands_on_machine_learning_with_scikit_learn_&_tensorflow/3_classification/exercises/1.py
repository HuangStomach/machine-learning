from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist['data'], mnist['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

param_grid = [
    {
        'weights': ['uniform', 'distance'], 
        'n_neighbors': [5, 25, 125, 625]
    }
]

knn_base = KNeighborsClassifier()
grid_search = GridSearchCV(knn_base, param_grid, cv = 3, verbose = 2, n_jobs = -1)
grid_search.fit(X_train, y_train)

knn = grid_search.best_estimator_
print(grid_search.best_params_)
print(knn.score(X_test, y_test))