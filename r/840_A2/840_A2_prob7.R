

#monitor:
#  - look at the time series plots of the generated sample and look for stationarity
#> autocorrelation function acf() ;; 1 bar is ok
#- Look at the marginal histograms		
#- monitor (plot) the estimator as the sample size increases and looks for the sign of convergence

#1. plot chain as time series	
#2. check that it is stationary using acf(), 1 bar ok
#3. Q7: marginal histograms
#4. plot estimator as n increases, does it converge?


# c

n = 10000
chain = matrix(ncol=2, nrow=n)
chain[1,] = c(1,1)

for (i in 2:n)
{
  x2_old = chain[i-1, 2]
  
  x1_new = rgamma(n=1, shape=3, rate=(x2_old^2 + 4))
  x2_new = rnorm(n=1, mean=(1/(x1_new + 1)), sd=(1 / (sqrt(2) * sqrt(x1_new + 1))))
  
  chain[i,] = c(x1_new, x2_new) 
}

par(mfrow=c(3,2))
# 1. plot chain as time series	
plot(chain[,1], type='l')
plot(chain[,2], type='l')
# 2. check that it is stationary using acf(), 1 bar ok
acf(chain[,1])
acf(chain[,2])
# 3. Q7: marginal histograms
hist(chain[,1])
hist(chain[,2])




# d

d = function(chain)
{
  x = chain[,1]
  y = chain[,2]
  return((x^2) * (y^3) * exp(-(x^2)))
}

# monitor theta_hat
theta_hat = rep(NA,n)
for (i in 2:n)
{
  theta_hat[i] = mean(d(chain[1:i,]))
}

# 4. plot estimator as n increases, does it converge?
par(mfrow=c(1,1))
plot(theta_hat, type='l')

# burn-in
burn = 1000
chain_b = chain[burn:n,]
mean(d(chain_b))


