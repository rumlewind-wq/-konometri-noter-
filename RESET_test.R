rm(list=ls(all=TRUE))
library(foreign)
library(AER)
library(lmtest)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/CARD.DTA")

l_1<-lm(lwage~educ+exper+black+south,data=Data)

yhatsq<-predict(l_1)^2
yhatcub<-predict(l_1)^3

l_2<-lm(lwage~educ+exper+black+south+yhatsq+yhatcub,data=Data)

linearHypothesis(l_2,c("yhatsq=0","yhatcub=0"))

#alternative : use the function resettest in the package lmtest
aux<-lm(lwage~educ+exper+black+south,data=Data)

r_test<-resettest(aux,2:3,c("fitted"),data=Data)


