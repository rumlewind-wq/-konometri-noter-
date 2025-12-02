rm(list=ls(all=TRUE))
library(foreign)
library(AER)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/MROZ.DTA")
Data<-Data[1:428,]

#H_o : E[z'u]=0
l_2sls<-ivreg(lwage~educ+exper+expersq|.-educ+motheduc+fatheduc+huseduc,data=Data)

#Sargan test
u_hat<-l_2sls$residuals

aux<-lm(u_hat~motheduc+fatheduc+huseduc+exper+expersq,data=Data)

LM_statistic<-summary(aux)$r.squared*dim(Data)[1]
p_value<-1-pchisq(LM_statistic,2)
#We do not reject H_0