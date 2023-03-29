
# 7

# a

d = function(x)
{
  return((x^3)*exp(x))
}

get_SE = function(x) # return variance of sample mean
{
  n = length(x)
  mu = sum(x) / n
  s_sq = sum((x - mu)^2) / ((n-1)*n) # notes p 30
  return(sqrt(s_sq))
}

get_CI = function(xbar, se, conf=.95)
{
  a = 1 - conf
  L = xbar + qnorm(a/2)*se
  U = xbar + qnorm(1 - a/2)*se
  return(c(L,U))
}

n = 100000
z = rnorm(n)

deltas = d(z)
theta_hat = mean(deltas)
SE = get_SE(deltas)
CI = get_CI(theta_hat, SE)
# whenever reporting a MC estimate, report n, SE of estimate, and CI
c(n, theta_hat, SE, CI)





# TEST - DONT INCLUDE
n = 100000
U1 = runif(n,0,1)
U2 = 1 - U1
#U2 = runif(n,0,1) 
z1 = qnorm(U1,0,1)
z2 = qnorm(U2,0,1)
hist(z1)
hist(-z2)
hist(z2)
cor(z1,z2)
cov(z1,z2)
var((z1 + z2) / 2)
var(z1)





# EXAMPLE 21

d = function(x)
{
  return(exp(x))
}

n = 100000
u1 = runif(n/2,0,1)
u2 = runif(n/2,0,1)

cov(d(u1), d(1-u1))
var(d(u1))
var((d(u1) + d(u2))/2)
var((d(u1) + d(1-u1))/2)


# EXAMPLE 22

get_SE = function(x) # return variance of sample mean
{
  n = length(x)
  mu = sum(x) / n
  s_sq = sum((x - mu)^2) / ((n-1)*n) # notes p 30
  return(sqrt(s_sq))
}

d = function(x)
{
  return(x / (2^x - 1))
}

d = function(z)
{
  return((z^3)*exp(z))
}

n = 100000

# standard MC
deltas_standard = d(rnorm(n))
mean(deltas_standard)
get_SE(deltas_standard)

# antithetic
z = rnorm(n/2)
deltas = d(c(z, -z))
mean(deltas)
get_SE((d(z) + d(-z))/2)
cor(d(z), d(-z))
1





# b

d = function(z)
{
  return((z^3)*exp(z))
}

# antithetic
n = 100000
z = rnorm(n/2)
deltas = (d(z) + d(-z))/2
theta_as = mean(deltas)
se_as = get_SE(deltas)
ci_as = get_CI(theta_as, se_as)
c(n, theta_as, se_as, ci_as)
cor(d(z), d(-z))
1





# c

nn = 1000
mc_simple = rep(0, nn)
mc_as = rep(0, nn)

for (i in 1:nn)
{
  n = 10000

  # standard
  z1 = rnorm(n)
  theta_hat = mean(d(z1))
  
  # antithetic
  z2 = rnorm(n/2)
  theta_as = mean((d(z2) + d(-z2))/2)
  
  mc_simple[i] = theta_hat
  mc_as[i] = theta_as
}

par(mfrow = c(2,2))
p1 = hist(mc_simple)
p2 = hist(mc_as)
plot( p1, col=rgb(0,0,1,1/4), xlim=c(0,10))
plot( p2, col=rgb(1,0,0,1/4), xlim=c(0,10), add=T)

c(var(mc_simple), var(mc_as))
# we see that the variances are about the same. in this case, we do not
# get any variance reduction using antithetic sampling because the 
# correlation between the estimators theta_1 and theta_2 is near 0. 


