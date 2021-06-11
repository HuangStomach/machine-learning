import Base
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

param_grid = [
    {
        'kernel': ['linear'], 
        'C': [10., 300., 3000., 30000.0]
    }
]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv = 5, return_train_score=True, n_jobs=-1)
grid_search.fit(Base.housing_prepared, Base.housing_labels)
svr = grid_search.best_estimator_
print(svr)

X_test = Base.strat_test_set.drop('median_house_value', axis=1)
y_test = Base.strat_test_set['median_house_value'].copy()
X_test_prepared = Base.full_pipeline.transform(X_test)
print(svr.score(X_test_prepared, y_test))
