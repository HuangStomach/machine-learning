'''
目标是根据乘客的年龄、性别、乘客等级、登船地点等属性来预测其是否存活。
登录Kaggle，进入Titanic挑战赛，下载train.csv和test.csv。把它们保存到datasets/titanic目录下。
'''
import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score

TITANIC_PATH = os.path.join("../datasets", "titanic")

class DataFrameSelector(BaseEstimator, TransformerMixin):
    '''
    过滤特定有价值的特征
    '''
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]

class MostFrequentImputer(BaseEstimator, TransformerMixin):
    '''
    用最多出现的特征进行填补
    '''
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series(
            [X[c].value_counts().index[0] for c in X],
            index=X.columns
        )
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)

def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)

train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")

'''
Survived：这是label，0意味着乘客没有存活，而1意味着他/她存活。
Pclass：乘客等级。
Name:
Sex:
Age:
SibSp：乘客在泰坦尼克号上有多少个兄弟姐妹和配偶。
Parch：乘客在泰坦尼克号上有多少孩子和父母。
Ticket：船票编号
Fare：支付的价格（以英镑计
Cabin：乘客的船舱号码
Embarked：乘客在哪里登上泰坦尼克号？

年龄、船舱和登船的属性有时是空的（少于891个非空），特别是船舱（77%是空的）。我们将暂时忽略机舱，而把注意力放在其他方面。年龄属性有大约19%的空值，所以我们需要决定如何处理它们。用年龄中位数替换空值似乎是合理的。

Name和Ticket属性可能有一些价值，但是要把它们转换成模型可以使用的有用的数字，会有点麻烦。所以现在，我们将忽略它们
'''

num_pipeline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ("imputer", SimpleImputer(strategy="median")),
])
cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False)),
])
preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
X_train = preprocess_pipeline.fit_transform(train_data)
y_train = train_data["Survived"]

from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10, n_jobs=-1)
print(forest_scores.mean())
