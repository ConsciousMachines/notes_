

d_RB = function(x)
{
  return(exp(9.5) * (x^2))
}

d_MC = function(x, e)
{
  return(exp(9) * (x^2) * (exp(e)))
}

get_SE = function(x) # return variance of sample mean
{
  n = length(x)
  mu = sum(x) / n
  s_sq = sum((x - mu)^2) / ((n-1)*n) # notes 2 p 30
  return(sqrt(s_sq))
}

get_CI = function(xbar, se, conf=.95)
{
  a = 1 - conf
  L = xbar + qnorm(a/2)*se
  U = xbar + qnorm(1 - a/2)*se
  return(c(L,U))
}


n = 100000000
x = rlnorm(n, 0, 1)
e = rnorm(n, 0, 1)
y = exp(9 + 3*log(x) + e)

# RB
deltas_RB = d_RB(x)
theta_hat_RB = mean(deltas_RB)
SE_RB = get_SE(deltas_RB)
CI_RB = get_CI(theta_hat_RB, SE_RB)

# simple
deltas_MC = d_MC(x, e)
theta_hat_MC = mean(deltas_MC)
SE_MC = get_SE(deltas_MC)
CI_MC = get_CI(theta_hat_MC, SE_MC)

# whenever reporting a MC estimate, report n, SE of estimate, and CI
# Rao Blackwell
round(c(n=n, theta=theta_hat_RB, se=SE_RB, ci=CI_RB), 0)
# Simple MC
round(c(n=n, theta=theta_hat_MC, se=SE_MC, ci=CI_MC), 0)
