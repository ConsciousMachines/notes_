
# 8

# a

get_SE = function(x) # return variance of sample mean
{
  n = length(x)
  mu = sum(x) / n
  s_sq = sum((x - mu)^2) / ((n-1)*n) # notes p 30
  return(sqrt(s_sq))
}

get_cov = function(x, y) # covariance of 2 sample means, p 30
{
  n = length(x)
  if (n != length(y))
  {
    stop('UNEQUAL LENGTH VECTORS')
  }
  xbar = sum(x)/n
  ybar = sum(y)/n
  diff1 = x - xbar
  diff2 = y - ybar
  cov_hat = sum(diff1 * diff2) / (n*(n-1))
  return(cov_hat)
}

get_CI = function(xbar, se, conf=.95)
{
  a = 1 - conf
  L = xbar + qnorm(a/2)*se
  U = xbar + qnorm(1 - a/2)*se
  return(c(L,U))
}

d1 = function(x)
{
  return(exp((-x^2)/2))
}


nn = 1000
tests = rep(0,nn)

lim_a = -5
lim_b = 5
real = sqrt(2 * pi) / (lim_b - lim_a)
# TODO: include derivation of why we multiply by (b-a)

for (i in 1:nn)
{
  n = 1000
  u1 = runif(n, lim_a, lim_b)
  
  # simple MC estimate
  deltas1 = d1(u1)
  theta_mc = mean(deltas1)
  
  # SE
  SE = get_SE(deltas1) #sqrt(var(d1(u1)) / n)
  
  # 95% confidence interval
  ci = get_CI(theta_mc, SE)
  
  if ((real > ci[1]) && (real < ci[2]))
  {
    tests[i] = 1
  }
}
sum(tests) # check that the real value falls in CI 95% of time




# b

d2 = function(x)
{
  return(1 - abs(x)/5)
}

nn = 50
hits1 = rep(0,nn)
hits2 = rep(0,nn)
test1 = rep(0,nn)
test2 = rep(0,nn)

for (i in 1:nn)
{
  n = 500
  
  u = runif(n, lim_a, lim_b)
  d1u = d1(u)
  d2u = d2(u)
  a = -cov(d1u, d2u) / var(d2u)
  deltas1 = d1u
  deltas2 = d1u + a*(d2u - 5/(lim_b-lim_a)) # integral is 5, /10 for (b-a)
  
  theta_mc = mean(deltas1)
  theta_cv = mean(deltas2)
  
  # SE
  SE1 = get_SE(deltas1)
  SE2 = get_SE(deltas2)
  ci1 = get_CI(theta_mc, SE1)
  ci2 = get_CI(theta_cv, SE2)
  if ((real > ci1[1]) && (real < ci1[2]))
  {
    hits1[i] = 1
  }
  if ((real > ci2[1]) && (real < ci2[2]))
  {
    hits2[i] = 1
  }
  
  test1[i] = theta_mc
  test2[i] = theta_cv
}
sum(hits1)
sum(hits2)

var(deltas1)
var(deltas2)
cov(d1(u), d2(u))

# average estimate
mean(test2)
# 0.2481642

# varaince of the estimate
var(test2) # TODO: is this the right thing? or write SE2?
# 4.235701e-05






# c

d3 = function(x)
{
  return(1 - (x^2)/25)
}

# an easy way to tell if it will be better is to see if its graph
# is more similar to the original function, in which case they will 
# have a higher correlation. 

# in part b, the function has the same overall slope: up and down.
# in part c, the function is farther away but has the same convexity
# in the center. it is hard to tell if it is a better fit. 
plot(u, d1(u), col='red')
points(u, d2(u), col='blue')

plot(u, d1(u), col='red')
points(u, d3(u), col='blue')


nn = 50
hits1 = rep(0,nn)
hits2 = rep(0,nn)
test1 = rep(0,nn)
test2 = rep(0,nn)

for (i in 1:nn)
{
  n = 500
  
  u = runif(n, lim_a, lim_b)
  d1u = d1(u)
  d2u = d3(u)
  a = -cov(d1u, d2u) / var(d2u)
  deltas1 = d1u
  deltas2 = d1u + a*(d2u - (20/3)/(lim_b-lim_a)) # integral is 20/3
  
  theta_mc = mean(deltas1)
  theta_cv = mean(deltas2)
  
  # SE
  SE1 = get_SE(deltas1)
  SE2 = get_SE(deltas2)
  ci1 = get_CI(theta_mc, SE1)
  ci2 = get_CI(theta_cv, SE2)
  if ((real > ci1[1]) && (real < ci1[2]))
  {
    hits1[i] = 1
  }
  if ((real > ci2[1]) && (real < ci2[2]))
  {
    hits2[i] = 1
  }
  
  test1[i] = theta_mc
  test2[i] = theta_cv
}
sum(hits1)
sum(hits2)

var(deltas1)
var(deltas2)
cov(d1(u), d2(u))

mean(test2)
# 0.2505198
var(test2)
# 0.0001110676

# conclusion: the variance reduction is worse with this function.











# example 20

d1 = function(x)
{
  return(exp(x))
}
d2 = function(x)
{
  return(x)
}

nn = 1000
hits1 = rep(0,nn)
hits2 = rep(0,nn)
test1 = rep(0,nn)
test2 = rep(0,nn)

real = exp(1) - 1

a = -(1 - (exp(1)-1)/2)*12 #1.690309#1.96032

for (i in 1:nn)
{
  n = 1000
  
  u = runif(n,0,1)
  d1u = d1(u)
  d2u = d2(u)
  #a = -cov(d1u, d2u) / var(d2u)
  deltas1 = d1u
  deltas2 = d1u + a*(d2u - 0.5)
  
  theta_mc = mean(deltas1)
  theta_cv = mean(deltas2)
  
  # SE
  SE1 = get_SE(deltas1)
  SE2 = get_SE(deltas2)
  ci1 = get_CI(theta_mc, SE1)
  ci2 = get_CI(theta_cv, SE2)
  if ((real > ci1[1]) && (real < ci1[2]))
  {
    hits1[i] = 1
  }
  if ((real > ci2[1]) && (real < ci2[2]))
  {
    hits2[i] = 1
  }
  
  test1[i] = theta_mc
  test2[i] = theta_cv
}
sum(hits1)
sum(hits2)

var(deltas1)
var(deltas2)
cov(d1(u), d2(u))


