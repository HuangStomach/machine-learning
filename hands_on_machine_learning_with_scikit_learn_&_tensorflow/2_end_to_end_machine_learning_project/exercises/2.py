import Base
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

param_grid = [
    {
        'kernel': ['linear'], 
        'C': [10., 300., 3000., 30000.0]
    }
]

svm_reg = SVR()
random_search = RandomizedSearchCV(svm_reg, param_grid, cv = 5, return_train_score=True, n_jobs=-1)
random_search.fit(Base.housing_prepared, Base.housing_labels)
svr = random_search.best_estimator_
print(random_search.cv_results_)
print(svr)

X_test = Base.strat_test_set.drop('median_house_value', axis=1)
y_test = Base.strat_test_set['median_house_value'].copy()
X_test_prepared = Base.full_pipeline.transform(X_test)
print(svr.score(X_test_prepared, y_test))
