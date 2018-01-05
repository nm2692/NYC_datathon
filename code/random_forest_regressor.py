import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import cross_val_score
df = pd.read_csv('~/Desktop/NYC Datathon Materials/train_data.csv')
df_test = pd.read_csv('~/Desktop/NYC Datathon Materials/test_data.csv')
# data cleaning process

Y = df['number_price'].values
df = df.drop('mean_price',1)
df = df.drop('number_price',1)
# Y = np.array([1 if y>=7 else 0 for y in Y])      #label class
X = df.as_matrix()
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std


Y_test = df_test['number_price'].values
df_test = df_test.drop('number_price',1)
df_test = df_test.drop('mean_price',1)
X_test = df_test.as_matrix()
mean_test = X_test.mean(axis=0)
std_test = X_test.std(axis=0)
X_test = (X_test - mean_test)/std_test









# # get scores of different numbers of decision trees
def get_boxplot(scores, trees, ylabel, title):
    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    bp = ax.boxplot(scores)
    ax.set_xticklabels(trees)
    plt.xlabel('Number of trees')
    plt.ylabel(ylabel)
    plt.ylim(0,1)
    plt.title(title)
    return
scores = []

for ne in range(70,250,10):
    clf = RandomForestRegressor(n_estimators = ne)
    score_list = cross_val_score(clf, X, Y, cv=10)
    scores.append(score_list)
#
get_boxplot(scores, range(70,250,10),'Regressor score','Regressor score as a function of the number of trees')
plt.show()



#
#
#
#
# repeat of previous but with F1 scoring instead

scores = []        # the second method to determine the best fitting number of decision trees
                    # after visulization of F1 scores as a function of the number of trees, we can decide the number of trees
                    # Finally, we can train our random forest using best number of trees

for ne in range(10,100,10):
    clf = RandomForestRegressor(n_estimators = ne)
    score_list = cross_val_score(clf, X, Y, cv=10, scoring='f1')
    scores.append(score_list)

get_boxplot(scores, range(10,100,10),'F1 score','F1 Scores as a function of the number of trees')
plt.show()


#plot feature importance

clf = RandomForestRegressor(n_estimators=40)

clf.fit(X,Y)
importance_list = clf.feature_importances_
name_list = df.columns
importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))
plt.barh(range(len(name_list)),importance_list,align='center')
plt.yticks(range(len(name_list)),name_list)
plt.xlabel('Relative Importance in the Random Forest')
plt.ylabel('Features')
plt.title('Relative importance of Each Feature')
plt.show()

#
#
#
# # search for good  using GridSearchCV
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import AdaBoostRegressor #For Regression
from sklearn.grid_search import GridSearchCV   #Perforing grid search
#
param_test1 = {'n_estimators':[20,30,40,50], 'max_depth':[3,6,8,12,24,32],
               'min_samples_split':[2,4,6],'min_samples_leaf':[1,2,4]}
gsearch1 = GridSearchCV(estimator = RandomForestRegressor(),
                        param_grid = param_test1, scoring='f1',n_jobs=4,iid=False, cv=5)


#enable out of bag error
param_test1 = {'n_estimators':[20,30,40,50], 'max_depth':[3,6,8,12,24,32],
               'min_samples_split':[2,4,6],'min_samples_leaf':[1,2,4]}
gsearch1 = GridSearchCV(estimator = RandomForestRegressor(),
                        param_grid = param_test1, scoring='f1',n_jobs=4,iid=False, cv=5)
print(gsearch1.fit(X,Y).best_params_)


#after we determine the best parameter, we can then plot the relative importance of each feature and of course train the model
clf = RandomForestRegressor(n_estimators=90,min_samples_leaf=2, min_samples_split=4,
                                      max_depth=4,oob_score=True)
# using the best parameters to train the model
print(clf.fit(X,Y))
importance_list = clf.feature_importances_
name_list = df.columns
importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))
plt.barh(range(len(name_list)),importance_list,align='center')
plt.yticks(range(len(name_list)),name_list)
plt.xlabel('Relative Importance in the Random Forest')
plt.ylabel('Features')
plt.title('Relative importance of Each Feature')
plt.show()


#
# #final step: plot decision surface using the two most important features
def plot_decision_surface(clf, X_train, Y_train):
    plot_step = 0.1

    if X_train.shape[1] != 2:
        raise ValueError("X_train should have exactly 2 columnns!")

    x_min, x_max = X_train[:, 0].min() - plot_step, X_train[:, 0].max() + plot_step
    y_min, y_max = X_train[:, 1].min() - plot_step, X_train[:, 1].max() + plot_step
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))

    clf.fit(X_train, Y_train)
    if hasattr(clf, 'predict_proba'):
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    else:
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.Reds)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=Y, cmap=plt.cm.Paired)
    plt.show()


imp_cols = clf.feature_importances_.argsort()[::-1][0:2]
X_imp = X[:, imp_cols]

plt.title('Random Forest classifier')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plot_decision_surface(clf, X_imp, Y)