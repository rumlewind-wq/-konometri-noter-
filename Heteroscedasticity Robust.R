# Heteroscedasticity robust

rm(list=ls())

# load data
library(haven)
wage1 <- read_dta("H:/Teaching/CBS/BA-BMECV1031U/2025/Lectures/A2 OLS topics/wage1.dta")

# generating missing variables:
wage1$marrmale <- wage1$married*(1-wage1$female)
wage1$marrfem <- wage1$married*wage1$female
wage1$singfem <- (1-wage1$married)*wage1$female

# regression with "normal" standard errors:
reg1 <- lm(lwage~ marrmale + marrfem + singfem + educ + exper + expersq + tenure + tenursq, data=wage1)
summary(reg1)

# regression with heteroscedasticity robust standard errors:
library(lmtest)
library(car)
#refined White robust SE:
coeftest(reg1, vcov=hccm)
#or classical White robust SE:
coeftest(reg1, vcov=hccm(reg1, type="hc0"))


