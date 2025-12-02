# Multiple regression

rm(list=ls())

# load data
library(haven)
wage1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A1 OLS/WAGE1.DTA")

# estimate model:
reg1 <- lm(lwage ~ educ + exper + tenure, data=wage1)
summary(reg1)

# compute VIF (Variance Inflator Factor) for all regressors

vif(reg1)