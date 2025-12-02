rm(list=ls(all=TRUE))
library(foreign)
library(AER)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2024/Lectures/B1 Endogeneity/CARD.DTA")

#test for the validity of the instrument nearc4
t<-lm(educ~nearc4+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669,data=Data)
#IV regression with nearc4 as IV for educ
l_iv<-ivreg(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669|nearc4+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669,data=Data)
#simple OLS
l<-lm(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669,data=Data)


summary(t)
summary(l_iv)
summary(l)


# Heteroskedasticity robust inference after IV:
coeftest(l_iv, vcov=vcovHC(l_iv, type="HC0")) 