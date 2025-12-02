# Newey West Standard Errors

rm(list=ls())

# load data
library(haven)
phillips <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/phillips.dta")

library(dynlm)
tsdata <- ts(phillips, start=1948)

# OLS estimation:
reg1 <- dynlm(inf~ unem, data=tsdata, end=1996)
summary(reg1)


# OLS with robust SE (Newey-West correction):
#one needs to specify maximum order of autocorrelation 
#resulting SE are robust up to the chosen order
#They are also robust with respect to heteroskedasticity
library(sandwich)
coeftest(reg1, vcov=NeweyWest(reg1, lag=3))

# For comparison: OLS with heteroskedasticity robust SE:
coeftest(reg1, vcov=hccm)
