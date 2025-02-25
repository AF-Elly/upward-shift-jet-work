
setwd(dir = "/Users/dongyili/科研/RProject/") #设置工作路径
source('ECOF_V1.r')
Z<-readin('obssigall_20_40_N_1980-2019_5yrmean_anom.dat','noise1_20_40_N_5yrmean.dat','noise2_20_40_N_5yrmean.dat')

t=seq(1980, 2019, 5)
plot(t, Z@Y, ylim=c(-1,1), type='l', xlab='Year', ylab='Mean u100-u200 Anomaly')
par(new=TRUE)
plot(t, Z@X, type='l', ylim=c(-1,1),xlab='', ylab='', col=2, main='20-40°N u100-u200')
legend(1968, 35, legend=c('OBS', 'SIG'), col=c(1,2), lty=c(1,1))
readline("Press <return to continue")

all_sig=matrix(scan('sigall_models_20_40_N_1980-2019_5yrmean_anom.dat'), byrow=T, nrow=28)
matplot(t, t(all_sig), ylim=c(-1,1), type='l', lty=2, xlab='Year', ylab='Mean u100-u200 Anomaly', col=7)
par(new=TRUE)
plot(t, Z@Y, ylim=c(-1,1), type='l', xlab='Year', ylab='Mean u100-u200 Anomaly')
par(new=TRUE)
plot(t, Z@X, type='l', ylim=c(-1,1),xlab='', ylab='', col=2, main='20-40°N u100-u200')
readline("Press <return to continue")


# nat_sig=matrix(scan('NAT_ann_1area.dat'), byrow=T, nrow=59)
# matplot(t, t(all_sig), ylim=c(-30,40), type='l', lty=2, xlab='Year', ylab='Mean Prep Anomaly (mm)', col=7)
# par(new=TRUE)
# matplot(t, t(nat_sig), ylim=c(-30,40), type='l', lty=3, xlab='Year', ylab='', col=4)
# par(new=TRUE)
# plot(t, Z@Y, ylim=c(-30,40), type='l', xlab='Year', ylab='Mean Prep Anomaly (mm)', lwd=2)
# par(new=TRUE)
# plot(t, Z@X, type='l', ylim=c(-30,40),xlab='', ylab='', col=2, main='High latitude precipitation', lwd=2)
# readline("Press <return to continue")

#              nsig是一个向量，若nsig=(158,200,160), 其意义为：三个强迫因子（信号）作用下的多元线性回归，158/200/160分别代表模式集合平均的成员数
#              文霞的例子中，nsig=（158），只评估了一个强迫因子（all）作用下的信号
#    o1.ols<-ols(Zr@Y,Zr@X,Zr@noise1,Zr@noise2,nsig=c(30,30,30)) 同时评估3个因子，nsig代表每个因子的样本数

u=redop(8,1,6)
Zr=redECOF(Z,u,8,1,timefirst=T)
 o1.ols<-ols(Zr@Y,Zr@X,Zr@noise1,Zr@noise2,nsig=28)
 par(mfrow=c(1,2))
 plotbetas(o1.ols)
 plotrstat(o1.ols)
 readline("Press <return to continue")

 o1.tls<-tls.A03(Zr@Y,Zr@X,Zr@noise1,Zr@noise2,nsig=28)
 par(mfrow=c(1,2))
 plotbetas(o1.tls)
 plotrstat(o1.tls)
 readline("Press <return to continue")

o1.rof<-tls.ROF(Zr@Y,Zr@X,Zr@noise1,Zr@noise2,nsig=28)
par(mfrow=c(1,2))
plotbetas(o1.rof)
plotrstat(o1.rof)
readline("Press <return to continue")

