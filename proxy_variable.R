rm(list=ls(all=TRUE))
#R package which permits the import of .dta files
library(foreign)
MyData <- read.dta("H:/Teaching/CBS/BA-BMECV1031U/2020/Lectures/B1 Endogeneity/WAGE2.DTA")

#linear regression without proxy
l<-lm(lwage~educ+exper+tenure+married+black+south+urban,data=MyData)
#linear rergression with proxy IQ for ability
l_IQ<-lm(lwage~educ+exper+tenure+married+black+south+urban+IQ,data=MyData)

summary(l)
summary(l_IQ)