# Variable Selection Linear Models
feature selection for linear regression models based on several statistical methods.
1. Pearson correlation method: select explanatory variable that a highly correlated with the response variable. In case explanatory variables are highly correlated, drop variables to reduce the problem of multicollinearity.
2. Bayesian information criterion backward reduction method: drop explanatory variables from set of regressors in order to minimize the BIC value.
3. P-value backward reduction method: drop explanatory variables from set of regressors based on the p-values. 
4. Lasso regression: small coefficients are set to zero. 
5. Ridge regression: small coefficients are set to zero. 
6. Regression tree: the relevance of explanaatory variables are based on presence in the single regression tree. 
7. Random forest: the relevance of explanaatory variables are based on presence in the random forest. 

\These 7 methods are combined to construct a subset of relevant explantory variables.
The performance is tested using a simulated dataset and the Boston housing dataset. 
