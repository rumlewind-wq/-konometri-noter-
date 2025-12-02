rm(list=ls(all=TRUE))

library(foreign)
library(AER)

Data<-read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/wage2.dta")
#OLS
l<-lm(lwage~educ+exper+tenure+married+south+urban+black,data=Data)
#IV pour aborder l'erreur dans le problème des variables
l_IV<-ivreg(lwage~educ+exper+tenure+married+south+urban+black+IQ|educ+exper+tenure+married+south+urban+black+KWW,data=Data)
#reduced form
l_IQ<-lm(IQ~exper+tenure+married+south+urban+black+KWW,data=Data)

h_IQ<-predict(l_IQ)
l_h_IQ<-lm(h_IQ~educ+exper+tenure+married+south+urban+black,data=Data)