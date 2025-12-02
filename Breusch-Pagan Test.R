# Breusch Pagan Test for Heteroscedasticity

rm(list=ls())

# load data
library(haven)
hprice1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/A2 OLS topics/hprice1.dta")

# estimate model:
reg1 <- lm(price~ lotsize + sqrft + bdrms, data=hprice1)
summary(reg1)


# automatic BP test:

library(lmtest)
bptest(reg1)


# manual regression of squared residuals:

reg2 <- lm(resid(reg1)^2~ lotsize + sqrft + bdrms, data=hprice1)
summary(reg2)
rsquared <- summary(reg2)$r.squared

# calculating of LM test statistic
LM = rsquared*length(reg2$residuals) 
LM

# p value
1-pchisq(LM,3)
