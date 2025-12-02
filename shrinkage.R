rm(list=ls(all=TRUE))

# Example code for common shrinkage estimators: Lasso, Ridge, Group Lasso
# This is optional material for the course BA-BMECV1031U Econometrics

# Uncomment on first run
# install.packages("glmnet", repos = "http://cran.us.r-project.org")
# For more information see https://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html

# install.packages("gglasso", repos = "http://cran.us.r-project.org")


library(glmnet)
library(repmis)
library(gglasso)

set.seed(123)

# Generate example data 
u <- rnorm(100)
beta <- runif(20, -1, 2) # random coefficients 
beta <- pmax(beta, 0) # set negative to zero 
x = matrix(rnorm(100 *20), 100, 20)
y = x%*%beta+u # standard linear model 

# Linear regression 
# Regression with a weighted penalty term
### Lasso ###
# alpha=1 gives lasso
fit_lasso <- glmnet(x, y, alpha=1)
print(fit_lasso)
plot(fit_lasso, xvar = "lambda", label = TRUE)
plot(fit_lasso, xvar = "dev", label = TRUE)
# Model selection
cv.lasso <- cv.glmnet(x,y)
plot(cv.lasso)
# best model - Coefficient extraction 
best_lambda_lasso <- cv.lasso$lambda.min
best_model_lasso <- glmnet(x, y, alpha = 1, lambda = best_lambda_lasso)
coef(best_model_lasso)

### Ridge ###
# alpha=0 gives ridge 
fit_ridge <- glmnet(x, y, alpha=0)
print(fit_ridge)
plot(fit_ridge, xvar = "lambda", label = TRUE)
plot(fit_ridge, xvar = "dev", label = TRUE)
# Model selection
cv.ridge <- cv.glmnet(x,y)
plot(cv.ridge)
# best model - Coefficient extraction 
best_lambda_ridge <- cv.ridge$lambda.min
best_model_ridge <- glmnet(x, y, alpha = 0, lambda = best_lambda_ridge)
coef(best_model_ridge)

### Group Lasso ###
# Select or unselect groups of covariates/regressors
# in the following example genes are grouped into 5 groups and the Group Lasso 
# is used find out whether a diseases is linked to certain groups of genes.

# The example Data consist of colon tissue samples from 62 patients. y indicates tumor (1)
# or normal (-1). x is a 62x100 matrix (expanded from a 62x20 matrix) giving the 
# expression levels of 20 genes. 5 consecutive columns corresponds to a grouped gene. 
# for more information see https://cran.r-project.org/web/packages/gglasso/gglasso.pdf

# load data 
data(colon)

# define 20 groups 
group <- rep(1:20,each=5)

# group lasso 
m <- gglasso(x=colon$x,y=colon$y,group=group,loss="ls")

# plot results 
plot(m) # plots the coefficients against the log-lambda sequence
plot(m,group=TRUE) # plots group norm against the log-lambda sequence
plot(m,log.l=FALSE) # plots against the lambda sequence

# coefficients at specified lambdas
coef(m, s=c(0.02, 0.03, 0.04))

# There are not yet many examples of the  Group Lasso in Economics and Business.
# A relevant one is the study "Variable selection with group structure: exiting
# employment at retirement age - A competing risks quantile regression analysis"
# By Shuolin Shi and Ralf A.Wilke, 2020, in Empirical Economics.