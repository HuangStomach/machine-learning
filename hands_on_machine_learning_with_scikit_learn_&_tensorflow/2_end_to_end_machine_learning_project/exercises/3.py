import Base
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv = 5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(Base.housing_prepared, Base.housing_labels)
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)

class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y = None):
        self.feature_indices_ = np.sort(
            np.argpartition(np.array(self.feature_importances), -self.k)[-self.k:]
        )
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]

extra_pipeline = Pipeline([
    ('preparation', Base.full_pipeline),
    ('feature_selection', FeatureSelector(feature_importances, 5))
])

housing_prepared = extra_pipeline.fit_transform(Base.housing)
print(housing_prepared)
