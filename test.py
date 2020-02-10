import numpy as np
import pandas as pd
import sklearn

from sklearn.datasets import load_iris

data = load_iris()

X = data['data']
y = data['target']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()

dtc.fit(X_train, y_train)

from sklearn_ex.tree import DecisionTreeHelper

helper = DecisionTreeHelper(dtc)

path_infos = helper.get_decision_path_info(X_test)

helper.print_decision_path(path_infos[1], data['feature_names'])

similar_id = helper.find_similar_instance(X_test[1:2], X_train)

print(similar_id)

print(y_train[similar_id])

tree_info = helper.get_tree_info()

print('depth:', tree_info['depth'])
print('n_leaves:', tree_info['n_leaves'])

df_leaves = tree_info['leaves']

print(df_leaves.head())

print('----------')

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=3)

rfc.fit(X_train, y_train)

from sklearn_ex.ensemble import RandomForestHelper

helper = RandomForestHelper(rfc)

df_trees_info, df_leaves_info_list = helper.get_forest_info()

print(df_trees_info.head())

for df_leaves_info in df_leaves_info_list:
    print(df_leaves_info.head())

similar_idxs, similar_nums = helper.get_similar_samples(X_test, X_train)

print(similar_idxs)

print(similar_nums)
