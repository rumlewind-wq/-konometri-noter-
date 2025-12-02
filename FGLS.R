# FGLS

rm(list=ls())

# load data
library(haven)
smoke <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/smoke.dta")

# OLS regression:

reg1 <- lm(cigs~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke)
summary(reg1)

yhat <- sum(reg1$fitted.values <=0)
yhat
# percentage of observations with fitted value <0
yhat/807


# Breusch- Pagan test for heteroscedasticity: 

reg1 <- lm(cigs~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke)
summary(reg1)

# automatic BP test:
library(lmtest)
bptest(reg1)

# manual regression of squared residuals:
reg2 <- lm(resid(reg1)^2~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke)
summary(reg2)
rsquared <- summary(lm(resid(reg1)^2~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke))$r.squared
# calculating of LM test statistic
LM = rsquared*807
LM
# p value
1-pchisq(LM,6)


# estimate by FGLS

# FGLS: estimation of the variance function:
logu2 <- log(resid(reg1)^2)
varreg <- lm(logu2~ lincome + lcigpric + educ + age + agesq + restaurn, data=smoke)

# FGLS: run WLS:
w <- 1/exp(fitted(varreg))

WLS <- lm(cigs~ lincome + lcigpric + educ + age + agesq + restaurn, weight=w, data=smoke)
summary(WLS)

