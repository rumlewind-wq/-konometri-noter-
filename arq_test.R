rm(list=ls(all=TRUE))

# for read.dta
library(foreign)

# for mutate 
library(dplyr)

# for bgtest
library(lmtest)

# setwd(".")
setwd("H:/Teaching/CBS/BA-BMECV1031U/2025/Lectures/A2 OLS topics")

phillips <- read.dta("phillips.dta")

### F-/LMTest for presence of AR(3)

phillips.limited <- subset(phillips, year<=1996)

reg <- lm(inf ~ unem, data=phillips.limited)
# obtain OLS residuals 
phillips.limited$res <- reg$residuals 

# F-test without constant in AR(3)model 
# generate lags 
phillips.limited <- phillips.limited %>%
  mutate(res_1 = dplyr::lag(res, n=1, default = NA)) %>%
  mutate(res_2 = dplyr::lag(res, n=2, default = NA)) %>%
  mutate(res_3 = dplyr::lag(res, n=3, default = NA)) 

ar3 <- lm(res ~ res_1 + res_2 + res_3 -1, data=phillips.limited)
summary(ar3)
# observe the reduced number of observations.
# F statistic has (3,43) d.f. and is >20.
# strong evidence of serial correlation of order 3.

# LM version of the test (special case of Breusch-Godfrey):
LM <- summary(ar3)$r.squared*(summary(ar3)$df[1]+summary(ar3)$df[2])
pvalue <- pchisq(LM, 3, lower.tail=FALSE)

message("Breusch-Godfrey statistic: LM = ", LM,", p-value = ", pvalue)

# Breusch Godfrey, 3rd order serial correlation 
bgtest(inf ~ unem, order = 3, type = "Chisq", data = phillips.limited)



