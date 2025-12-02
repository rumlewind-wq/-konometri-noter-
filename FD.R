# FD model

rm(list=ls())

# load data
library(haven)
intdef <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/intdef.dta")

# define yearly time series beginning in 1948
tsdata <- ts(intdef, start=1948)
library(dynlm)

reg1 <- dynlm(i3~ inf + def, data=tsdata)
summary(reg1)

# compute OLS residuals:
residuals1 <- resid(reg1)

# t-test without constant in AR(1)model
library(lmtest)
coeftest(dynlm(residuals1~ L(residuals1) + 0))

# evidence for positive serial correlation in errors


# first differenced model:
reg2 <- dynlm(d(i3)~ d(inf) + d(def) + 0, data=tsdata)
summary(reg2)

# compute OLS residuals in differenced model
residuals2 <- resid(reg2)

# t-test without constant in AR(1)model
coeftest(dynlm(residuals2~ L(residuals2) + 0))


#################################################
# Remark
#################################################
# comparing the results from the two regressions:
# the coefficients in the level model appear to have a plausible sign and they are significant
# the coefficients in the differenced model are insignificant.
# the coefficients vary statistically across the models:
# evidence for violation of the Gaus-Maarkov assumptions in the level model (exogeneity)



