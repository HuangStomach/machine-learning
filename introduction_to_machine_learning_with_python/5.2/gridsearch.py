from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

iris = load_iris()
param_grid = [
    {
        'kernel': ['kbf'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 10, 100]
    },
    {
        'kernel': ['linear'],
        'C': [0.001, 0.01, 0.1, 1, 10, 100]
    }
]
scores = cross_val_score(
    GridSearchCV(SVC(), param_grid, cv=5, n_jobs=-1),
    iris.data, iris.target, cv=5, n_jobs=-1
)
print(scores)
