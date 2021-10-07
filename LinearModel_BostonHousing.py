## Lasso regression
URL = 'https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b'

## Linear Regression
URL = 'https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155'

## Regression Tree
URL = 'https://scikit-learn.org/stable/modules/tree.html'
URL = 'https://www.geeksforgeeks.org/python-decision-tree-regression-using-sklearn/'

## Random Forest
URL = 'https://mljar.com/blog/feature-importance-in-random-forest/'

## Ridge regression
URL = 'https://machinelearningmastery.com/ridge-regression-with-python/'

## Statistical tests
URL = 'https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/'
URL = 'https://www.statsmodels.org/devel/examples/notebooks/generated/regression_diagnostics.html'

## Tuning Ridge Parameters
URL = 'https://machinelearningmastery.com/ridge-regression-with-python/'

## Variable Selection Linear Model
URL = 'https://towardsdatascience.com/feature-selection-with-pandas-e3690ad8504b'



## Packages
from scipy.stats.stats import SigmaclipResult
from sklearn import datasets
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as lm
import math as m

from linearmodels import IV2SLS, IVLIML, IVGMM, IVGMMCUE
from scipy import stats
from scipy.stats import f
from scipy.stats import t
from scipy.stats import chi2
from scipy.stats import ttest_ind
from scipy.stats import shapiro
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split
from sklearn import tree
from statsmodels.stats.diagnostic import het_white
from statsmodels.compat import lzip
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import linear_rainbow
from statsmodels.stats.diagnostic import het_breuschpagan

def SimulationModel(n,m,v):
    #n,m = 1000, 20
    k = round(m/10)
    var_names = ['x'+str(i) for i in range(1,m+1)]
    var_names.append('u')
    ## Bernouli distribution
    x1 = np.random.binomial(1, 0.5, size=(n, k))    ## Equal prop 1 and 0
    x2 = np.random.binomial(1, 0.2, size=(n, k))    ## lower prop 1 and 0
    ## Normal distribution
    x3 = np.random.normal(0, 1, size=(n, k))        ## low variance
    x4 = np.random.normal(0, 5, size=(n, k))        ## high variance
    ## Exponential distribution
    x5 = np.random.exponential(1, size=(n, k))      ## low variance
    x6 = np.random.exponential(5, size=(n, k))      ## high variance
    ## Uniform distibrution   
    x7 = np.random.uniform(0, 10, size=(n,k))       ## low variance
    x8 = np.random.uniform(0, 10, size=(n,k))       ## high variance
    ## Normal distribution
    x9 = np.random.normal(0, 1, size=(n, k))        ## low variance
    x10 = np.random.normal(0, 5, size=(n, k))       ## high variance
    u = np.random.normal(0, 2, size=(n, 1))        
    X = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,u), axis=1)
    weight_vec = v

    df = pd.DataFrame(X, columns=var_names)
    weight = pd.DataFrame(pd.Series(weight_vec, index=var_names, name='MEDV'))
    df = pd.concat([df,df.dot(weight)],axis=1)
    df = df.drop('u', 1)
    return df

def RelevantVariablesSimulation(v):
    rel_var = []
    for i in range(len(v)):
        if v[i] != 0:
            rel_var.append('x'+str(i+1))
    return rel_var[0:len(rel_var)-1]

def BostonData():
    boston_dataset = load_boston()
    #print(boston_dataset.DESCR)
    df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
    #df.head()
    df["MEDV"] = boston_dataset.target
    X = df.drop("MEDV",1)   #Feature Matrix
    y = df["MEDV"]          #Target Variable
    #print(df.head())
    #print(df.isnull().sum())
    return df

def DataChoice(ch):
    if ch == 'Simulation':
        df = SimulationModel(n,m,v)
    if ch == 'Boston':
        df = BostonData()
    return df 

def ScattterPlots():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    features = list(X.columns)
    plt.figure()
    for i in range(0,len(features)):
        plt.subplot(2, m.ceil(len(features)/2) , i+1)
        x = df[features[i]]
        plt.scatter(x, y, marker='o')
        plt.xlabel(features[i])
    plt.title('K')
    plt.show()
    return features


def OLS():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    X = lm.add_constant(X)
    model = lm.OLS(y,X).fit()
    return model.summary()

#print(OLS())


#####   Variable seletion
#####   1. Peason Correlation method:
#####       only keep variables with high correlation with target
#####       drop features with extreme high correlation with other feature
#####   2. BIC criterion method:
#####       drop features in case of improvement of BIC criterion
#####   3. P-value reductiono method:
#####       drop features based on p-values
#####   4. Lasso regression:
#####       small coefficients are set to zero
#####   5. Ridge regression:
#####       small coefficients are set to zero
#####   6. Regression tree:
#####       relevance based on presence in the tree
#####   7. Random forest:
#####       relevance based on presence in the random forest
#####   8. Combin the 7 methods: 
#####       a combination of the 7 methods result in a subset of variables
#####   9. Variance Inflation Factor (VIF) for multicollinearity 
#####       drop a variable if the VIF value exceeds critical value


def PearsonCorrelation():
    df = DataChoice(ch)
    ## find variables with covariance greater than 0.3 with dependent variable
    cor = df.corr().round(2)
    sns.heatmap(cor, annot=True)
    #plt.show()
    cor_target = abs(cor["MEDV"])
    rvar = cor_target[cor_target>0.3]
    rvar_list = list(rvar.index)
    while len(rvar_list) > 0:
        rvar_df = df[rvar_list]
        cor_new = rvar_df.corr()
        high_cor = []
        cor_mat = cor_new.to_numpy()
        for i in range(0,len(cor_mat)):
            for j in range(i,len(cor_mat[i])):
                if abs(cor_mat[i][j]) > 0.8 and i != j and (j != len(cor_mat)-1 and i != len(cor_mat)-1):
                    high_cor.append([rvar_list[i],rvar_list[j]])
        ## count presence of features in pairs 
        rvar_list_count = []
        for i in range(0,len(rvar_list)):
            rvar_list_count.append([rvar_list[i],0])
        for i in range(0,len(high_cor)):
            for j in range(0,len(high_cor[i])):
                for k in range(0,len(rvar_list)):
                    if rvar_list[k] == high_cor[i][j]:
                        rvar_list_count[k][1] += 1
        rvar_trans = np.array(rvar_list_count, dtype=object).transpose()
        highest_index = np.argmax(rvar_trans[1])
        ## If count of high correlation is larger than 1, drop this feature
        if np.amax(rvar_trans[1]) > 1:
            rvar_list.pop(highest_index)
        ## If count equals one, keep the feature with highest correlation with target
        elif np.amax(rvar_trans[1]) == 1:
            for v in range(0,len(rvar_list)):
                for w in range(v,len(rvar_list)):
                    if cor_mat[v][len(rvar_list)-1] < cor_mat[w][len(rvar_list)-1]:
                        highest_index = v
                        rvar_list.pop(v)
        ## If no pairs with high correlation remain, stop loop
        elif np.amax(rvar_trans[1]) == 0:
            break
    rvar_list.pop(len(rvar_list)-1)
    return rvar_list

def BackwardElimination():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    X = lm.add_constant(X)
    model = lm.OLS(y,X).fit()
    rvar_list = list(X.columns)
    start_bic = model.bic
    #for i in range(0,10):
    while len(rvar_list) > 0:
        rvar = rvar_list.copy()
        alt_bic = [(y.shape[0])**2]
        for i in range(1,len(rvar)):
            X_alt = X[rvar]
            X_alt = X_alt.drop(X_alt.columns[i], axis=1)
            model_alt = lm.OLS(y,X_alt).fit()
            alt_bic.append(model_alt.bic)
        if np.amin(alt_bic) < start_bic:
            min_index = np.argmin(alt_bic)
            rvar_list.pop(min_index)
            start_bic = np.amin(alt_bic)
        elif np.amin(alt_bic) > start_bic:
            break
    return rvar_list[1:len(rvar_list)]

def BackWardpvalue():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    X = lm.add_constant(X)
    model = lm.OLS(y,X).fit()
    rvar_list = list(X.columns)
    while len(rvar_list) > 0:
        rvar = rvar_list.copy()
        X_alt = X[rvar]
        model_alt = lm.OLS(y,X_alt).fit()
        pv = model_alt.pvalues.values
        max_index = np.argmax(pv)
        if np.amax(pv) > 0.1:
            rvar_list.pop(max_index)
        elif np.max(pv) <= 0.1:
            break
    return rvar_list

def Lasso():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    reg = LassoCV()
    reg.fit(X, y)
    coef = pd.Series(reg.coef_, index = list(X.columns))
    rel_var = []
    for i in range(0,len(list(X.columns))):
        if abs(coef[i]) > 0.05:
            rel_var.append(list(X.columns)[i])
    return rel_var

def Ridge():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    reg = RidgeCV()
    reg.fit(X, y)
    coef = pd.Series(reg.coef_, index = list(X.columns))
    rel_var = []
    for i in range(0,len(list(X.columns))):
        if abs(coef[i]) > 0.05:
            rel_var.append(list(X.columns)[i])
    return rel_var

def Tree():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    feature_names=list(X.columns)
    tree_model = DecisionTreeRegressor(max_depth=4, random_state=1234)
    tree_fit = tree_model.fit(X, y)  
    text_representation = tree.export_text(tree_model,feature_names=list(X.columns))
    fh = open("decistion_tree.log", "w+")
    fh.write(text_representation)
    fh = open("decistion_tree.log", "r")
    raw_data = fh.readlines()
    text_tree = []
    for line in raw_data:
        line = line.rstrip('\n').split()
        text_tree.append(line)
    rel_features = []
    for i in range(0,len(text_tree)):
        for j in range(0,len(text_tree[i])):
            for k in range(0,len(feature_names)):
                if feature_names[k] == text_tree[i][j]:
                    rel_features.append(feature_names[k])
    rel_features= list(dict.fromkeys(rel_features))
    tree_plot = tree.plot_tree(tree_model, filled=True,feature_names=list(X.columns))
    #plt.show()
    fig = plt.figure()
    return rel_features

def RandomForest():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    feature_names=list(X.columns)
    rf = RandomForestRegressor(n_estimators=500)
    rf.fit(X, y)
    sorted_idx = rf.feature_importances_.argsort()
    sorted_feature_names = [feature_names[i] for i in sorted_idx]
    plt.barh(sorted_feature_names, rf.feature_importances_[sorted_idx])
    plt.xlabel("Random Forest Feature Importance")  
    #plt.show()
    #return [rf.feature_importances_[sorted_idx],sorted_feature_names]
    sorted_feature_importance = rf.feature_importances_[sorted_idx]
    x = []
    for i in range(0,len(sorted_feature_names)):
        if sorted_feature_importance[i] > 0.02:
            x.append(sorted_feature_names[i])
    return x

def VariableSelection():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    r1,r2,r3,r4,r5,r6,r7 = PearsonCorrelation(), BackwardElimination(), BackWardpvalue(), Lasso(), Ridge(), Tree(), RandomForest()
    mat_var = [r1,r2,r3,r4,r5,r6,r7]
    features = list(X.columns)
    rvar_list_count = []
    for i in range(0,len(features)):
        rvar_list_count.append([features[i],0])
    for i in range(0,len(mat_var)):
        for j in range(0,len(mat_var[i])):
            for k in range(0,len(features)):
                if features[k] == mat_var[i][j]:
                    rvar_list_count[k][1] += 1
    com_rel_var = []
    for i in range(0,len(rvar_list_count)):
        if rvar_list_count[i][1] > 3:
            com_rel_var.append(rvar_list_count[i][0])
    return com_rel_var


def Vif():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    rel_var = VariableSelection()
    Xr = X[rel_var]
    Xr = lm.add_constant(Xr)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [round(variance_inflation_factor(Xr.values, i),2) for i in range(Xr.shape[1])]
    vif["features"] = Xr.columns
    vif_array = np.array(vif, dtype=object)
    vif_rel_var = []
    for i in range(1,len(vif_array)):
        if vif_array[i][0] < 10:
            vif_rel_var.append(vif_array[i][1])
    return vif_rel_var

#####   Regression estimation after variable selection
#####   OLS vs GLS, use White test 
#####   Shapiro test for normality errors

def Shapiro():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    rel_var = Vif()
    Xr = X[rel_var]
    Xr = lm.add_constant(Xr)
    model = lm.OLS(y,Xr).fit()
    residuals = np.array(model.resid, dtype=object)
    name = ['Statistic','p value']
    test = shapiro(residuals)
    return lzip(name, test)

def Rainbow():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    rel_var = Vif()
    #rel_var = ['CRIM', 'ZN', 'CHAS', 'NOX', 'RM', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
    Xr = X[rel_var]
    Xr = lm.add_constant(Xr)
    model = lm.OLS(y,Xr).fit()
    name = ['f value', 'p value']
    test = linear_rainbow(model)
    ypred = model.predict(Xr)
    return lzip(name, test)

def BreuschPagan():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    rel_var = VariableSelection()
    Xr = X[rel_var]
    Xr = lm.add_constant(Xr)
    model = lm.OLS(y,Xr).fit()
    test = het_breuschpagan(model.resid, model.model.exog)
    return round(test[3],2)

def GLS():
    df = DataChoice(ch)
    y,X = df["MEDV"], df.drop("MEDV",1)
    rel_var = VariableSelection()
    Xr = X[rel_var]
    Xr = lm.add_constant(Xr)
    pBP = BreuschPagan()
    if pBP < 0.05:
        model = lm.OLS(y,Xr).fit(cov_type='HC1')
    elif pBP >= 0.05:
        model = lm.OLS(y,Xr).fit()
    return model.summary()

## n number of obersvations in simulation dataset
n = 1000
## m number of features in simulation dataset
m = 20
## coefficientens of featurs
v = [5,0,-3,0,2,0,-4,0,2,0,-1,0,6,0,1,0,0,0,0,0,1]
## 'Simulation' for using simulation data
## 'Boston' for using Bosting housing data
ch = 'Simulation'

print(PearsonCorrelation())
print(BackwardElimination())
print(BackWardpvalue())
print(Lasso())
print(Ridge())
print(Tree())
print(RandomForest())
print(VariableSelection())
print(GLS())