rm(list=ls(all=TRUE))
library(foreign)
library(AER)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/CARD.DTA")
#delete observations with missing values for fatheduc
Data<-Data[!is.na(Data$fatheduc),]
# 2sls with nearc4 and fatheduc as instruments for educ
l_2sls<-ivreg(lwage~educ+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669|.-educ+nearc4+fatheduc,data=Data)

#determine p
u_hat<-l_2sls$residuals

aux<-lm(u_hat~nearc4+fatheduc+exper+expersq+black+smsa+south+smsa66+reg662+reg663+reg664+reg665+reg666+reg667+reg668+reg669,data=Data)

LM_statistic<-summary(aux)$r.squared*dim(Data)[1]
p_value<-1-pchisq(LM_statistic,1)
