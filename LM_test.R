# LM test
# Here we test for two exclusion restrictions

rm(list=ls())

# load data
library(haven)
wage1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Lectures/A1 OLS/wage1.dta")

# estimate model:
reg1 <- lm(lwage~ educ + exper, data=wage1)
summary(reg1)


# auxiliary regression:

reg2 <- lm(resid(reg1)~ educ + exper + tenure + married, data=wage1)
summary(reg2)


# calculating the LM test statistic
LM <- summary(reg2)$r.squared*526
LM

# p value
1-pchisq(LM,2)
#H_0 clearly rejected
