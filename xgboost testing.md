## 模型测试

在上面预处理过的数据的基础上，我们来初步对一系列常用分类模型进行测试。 

我们将利用 python 来进行测试。 

首先我们使用 pandas 从 csv 文件中导入数据，标签，再import一些备用的库，为后续测试备用。

```python
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import KFold
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics  
from sklearn.grid_search import GridSearchCV   
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
def load_data():
    X = pd.read_csv('xtrain.csv')
    X = np.array(X)
    y = pd.read_csv('label.csv')
    y = np.array(y).transpose()[0,:]
    xtest = pd.read_csv('xtest.csv')
    shot_id = xtest["shot_id"]
    xtest = np.array(xtest.drop("shot_id",axis=1))
    return X,y,xtest,shot_id
```

（1）用xgboost计算各个特征的重要性，同时计算AUC值。

```python
def modelfit(alg, dtrain, dtest, predictors, useTrainCV=True, cv_folds=10, early_stopping_rounds=50): 
  if useTrainCV: 
    xgb_param = alg.get_xgb_params()
    xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain["shot_made_flag"]。values)
    xgtest = xgb.DMatrix(dtest[predictors].values)
    cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()["n_estimators"], nfold=cv_folds,metrics="auc", early_stopping_rounds=early_stopping_rounds)
    alg.set_params(n_estimators=cvresult.shape[0])
#Fit the algorithm on the data
alg.fit(dtrain[predictors], dtrain["shot_made_flag"],eval_metric="auc")
#Predict training set:
dtrain_predictions = alg.predict(dtrain[predictors])
dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
#Print model report:
print ("\nModel Report")
print ("Accuracy : %.4g" % metrics.accuracy_score(dtrain["shot_made_flag"].values,
dtrain_predictions))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(dtrain["shot_made_flag"], dtrain_predprob))
feat_imp = pd.Series(dict(alg.get_booster().get_fscore())).sort_values(ascending=False).head(20)
print(feat_imp)
feat_imp.plot(kind="bar", title=Feature Importances)
plt.ylabel(Feature Importance Score)
plt.show()
```

画出前20个重要的特征：

![1566889437077](https://github.com/yhding1996/kobe-shooting-selection/blob/master/pp/feature.png?raw=true)

输出结果：

![1566889505739](https://github.com/yhding1996/kobe-shooting-selection/blob/master/pp/result.png?raw=true)

我们尝试使用前面较为重要的特征去训练模型，但是得出的效果都没有用所有特征训练的结果好，因
此最终我们还是选择了选择全部特征去训练模型。 

(2) 模型调参和优化
max_depth 与 min_child_weight 

```python
param_test1 = {
    "max_depth":[3,5,7],
    "min_child_weight":[1,3,5]
}
gsearch1 = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, n_estimators=10, max_depth=5,min_child_weight=1, gamma=0, subsample=0.8,colsample_bytree=0.8,objective= binary:logistic, nthread=4, scale_pos_weight=1, seed=27), param_grid = param_test1 , scoring="roc_auc",n_jobs=4,iid=False , cv=5)
gsearch1.fit(train[predictors],train["shot_made_flag"])
print([gsearch1.grid_scores_ , gsearch1.best_params_ , gsearch1.best_score_])
```



 输出结果：

```python
[[mean: 0.69016 , std: 0.00566 , params: {"max_depth": 3, "min_child_weight": 1},
mean: 0.69041 , std: 0.00591 , params: {"max_depth": 3, "min_child_weight": 3},
mean: 0.69050 , std: 0.00592 , params: {"max_depth": 3, "min_child_weight": 5},
mean: 0.70140 , std: 0.01276 , params: {"max_depth": 5, "min_child_weight": 1},
mean: 0.70170 , std: 0.01172 , params: {"max_depth": 5, "min_child_weight": 3},
mean: 0.70138 , std: 0.01230 , params: {"max_depth": 5, "min_child_weight": 5},
mean: 0.69254 , std: 0.01789 , params: {"max_depth": 7, "min_child_weight": 1},
mean: 0.69303 , std: 0.01580 , params: {"max_depth": 7, "min_child_weight": 3},
mean: 0.69433 , std: 0.01621 , params: {"max_depth": 7, "min_child_weight": 5}],
{"max_depth": 5, "min_child_weight": 3}, 0.7016990223660348]
```

n_estimators 与 gamma :

```python
param_test2 = {
    "n_estimators":[20,50,100,200,300,500,1000],
     gamma:[0.0,0.1,0.2,0.3]
}
gsearch2 = GridSearchCV(estimator = XGBClassifier(learning_rate=0.1, n_estimators=10, max_depth=5,
min_child_weight=3, gamma=0.4, subsample=0.8, colsample_bytree=0.8,
objective= binary:logistic, nthread=4, scale_pos_weight=1,seed=27),
param_grid = param_test2 , scoring="roc_auc",n_jobs=4,iid=False , cv=5)
gsearch2.fit(train[predictors],train["shot_made_flag"])

print(gsearch2.grid_scores_ , gsearch2.best_params_ , gsearch2.best_score_)
```

输出结果：

```python
[mean: 0.70156 , std: 0.01313 , params: {gamma: 0.0, "n_estimators": 20},
mean: 0.68861 , std: 0.02780 , params: {gamma: 0.0, "n_estimators": 50},
mean: 0.68149 , std: 0.02590 , params: {gamma: 0.0, "n_estimators": 100},
mean: 0.70154 , std: 0.01313 , params: {gamma: 0.1, "n_estimators": 20},
mean: 0.68834 , std: 0.02775 , params: {gamma: 0.1, "n_estimators": 50},
mean: 0.68129 , std: 0.02662 , params: {gamma: 0.1, "n_estimators": 100},
mean: 0.70149 , std: 0.01307 , params: {gamma: 0.2, "n_estimators": 20},
mean: 0.68818 , std: 0.02762 , params: {gamma: 0.2, "n_estimators": 50},
mean: 0.68102 , std: 0.02644 , params: {gamma: 0.2, "n_estimators": 100},
mean: 0.70120 , std: 0.01309 , params: {gamma: 0.3, "n_estimators": 20},
mean: 0.68836 , std: 0.02769 , params: {gamma: 0.3, "n_estimators": 50},
mean: 0.68158 , std: 0.02656 , params: {gamma: 0.3, "n_estimators": 100}],
{gamma: 0.0, "n_estimators": 20} 0.7015581986024912
```

综合上述结果，我们选择参数：
’gamma’: 0.0,
’n_estimators’: 10
’max_depth’: 5
’min_child_weight’: 3
因此，最终的模型代码为 

```python
xgbc = XGBClassifier(learning_rate=0.1,
                     n_estimators=10, # 树的个数--棵树建立10xgboost
                     max_depth=5, # 树的深度
                     min_child_weight = 3, # 叶子节点最小权重
                     gamma=0., # 惩罚项中叶子结点个数前的参数
                     subsample=0.8, # 随机选择80%样本建立决策树
                     colsample_btree=0.8, # 随机选择80%特征建立决策树
                     objective=binary:logistic, # 指定损失函数
                     scale_pos_weight=1, # 解决样本个数不平衡的问题
                     random_state=27 # 随机数
                      )
# modelfit(xgbc , train , xtest , predictors)
xgbc.fit(train[predictors],train["shot_made_flag"])
result = xgbc.predict_proba(xtest)
result = pd.DataFrame(result)
result.to_csv("~/Desktop/re.csv")
```

