# AR(1) t-test
# test for the presence of AR(1)

rm(list=ls())

# load data
library(haven)
phillips <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/phillips.dta")

# define yearly time series beginning in 1948
tsdata <- ts(phillips, start=1948)

library(dynlm)

# estimation of static Phillips curve:
reg1 <- dynlm(inf~ unem, data=tsdata, end=1996)
summary(reg1)

# compute OLS residuals:
residuals <- resid(reg1)

# t-test without constant in AR(1) model:
library(lmtest)
coeftest(dynlm(residuals~ L(residuals) + 0))

# t-test with constant in AR(1) model: 
coeftest(dynlm(residuals~ L(residuals)))

# heteroskedasticity robust version of t-test without constant in AR(1) model:
library(sandwich)
coeftest(dynlm(residuals~ L(residuals) + 0), vcovHAC)


# all tests suggest at 1% that there is evidence for positive serial correlation in errors
