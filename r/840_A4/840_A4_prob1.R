




em_algo = function(x, p, lam1, lam2)
{
  lik = -Inf
  
  while (TRUE)
  {
    # E step
    pois1 =     p * dpois(x, lam1)
    pois2 = (1-p) * dpois(x, lam2)
    w     = pois1 / (pois1 + pois2)
    
    # M step
    p    = mean(w)
    lam1 = sum(w     * x) / sum(w)
    lam2 = sum((1-w) * x) / sum(1-w)
    
    # check convergence 
    lik_new = sum(log(p * dpois(x, lam1) + (1-p) * dpois(x, lam2)))
    if (abs(lik_new - lik) < 1e-8)
      break
    
    lik = lik_new
  }
  return(c(p, lam1, lam2, lik_new))
}


grid_search = function(x) # search over a range of parameters
{
  df = matrix(nrow=20*20*9,ncol=4)
  colnames(df) = c('p', 'lam1', 'lam2', 'lik')
  i = 1
  for (p in seq(0.1, 0.9, 0.1))
  {
    for (lam1 in seq(1,20))
    {
      for (lam2 in seq(1,20))
      {
        df[i,] = em_algo(x, p, lam1, lam2)
        i = i + 1
      }
    }
  }
  max_idx = which.max(df[,4])
  return(df[max_idx,])
}

generate_data = function(n = 20, p = 0.3, lam1 = 5, lam2 = 15)
{
  x = rep(NA,n)
  for (i in 1:n)
  {
    lam = if (runif(1) < p) lam1 else lam2
    x[i] = rpois(1, lam)
  }
  return(x)
}

x = c(24, 18, 21, 5, 5, 11, 11, 17, 6, 7, 20, 13, 4, 16, 19, 21, 4, 22, 8, 17)
grid_search(x)
for (n in c(20,100,1000))
  print(grid_search(generate_data(n)))































p = 0.5
lam1 = 1
lam2 = 10

x = c(24, 18, 21, 5, 5, 11, 11, 17, 6, 7, 20, 13, 4, 16, 19, 21, 4, 22, 8, 17)

lik = -Inf

while (TRUE)
{
  # E step
  pois1 =     p * dpois(x, lam1)
  pois2 = (1-p) * dpois(x, lam2)
  w     = pois1 / (pois1 + pois2)
  
  # M step
  p    = mean(w)
  lam1 = sum(w     * x) / sum(w)
  lam2 = sum((1-w) * x) / sum(1-w)
  
  # check convergence 
  lik_new = sum(log(p * dpois(x, lam1) + (1-p) * dpois(x, lam2)))
  if (abs(lik_new - lik) < 1e-8)
    break
  
  lik = lik_new
}






expected_log_likelihood <- function(params, data, Z_probs) {
  p <- params[1]
  lambda1 <- params[2]
  lambda2 <- params[3]
  
  # Compute the Poisson probabilities for each component
  poisson1 <- dpois(data, lambda1)
  poisson2 <- dpois(data, lambda2)
  
  # Compute the expected log-likelihood using Z_probs
  log_likelihood <- sum(Z_probs * log(p * poisson1) + (1-Z_probs) * log((1 - p) * poisson2))
  
  return(log_likelihood)
}

# Vector of parameter estimates
params_hat <- c(p, lam1, lam2)

# Compute the Hessian matrix
H <- hessian(expected_log_likelihood, params_hat, data = x, Z_probs = w)

OIM <- -H

var_cov_matrix <- solve(OIM)

std_errors <- sqrt(diag(var_cov_matrix))

std_errors
1











library(numDeriv)

observed_lik = function(params)
{
  p = params[1]
  lam1 = params[2]
  lam2 = params[3]
  
  pois1 = dpois(x, lam1)
  pois2 = dpois(x, lam2)
  
  lik = sum(log(p*pois1 + (1-p)*pois2))
  
  return(lik)
}

# https://www.rdocumentation.org/packages/numDeriv/versions/2016.8-1.1/topics/hessian
H = hessian(observed_lik, c(p,lam1,lam2))

var_cov_matrix = solve(-H)

err = sqrt(diag(var_cov_matrix))

c(std_p=err[1], std_lam1=err[2], std_lam2=err[3])



















# hessian
s = sum(w)
n = length(x)
partial_pp = -(s/(p^2)) -((n-s)/((1-p)^2))
partial_11 = (-1/(lam1^2)) * sum(w*x)
partial_22 = (-1/(lam2^2)) * sum((1-w)*x)



hess_inv = matrix(data=0,nrow=3,ncol=3)
hess_inv[1,1] = 1/partial_pp
hess_inv[2,2] = 1/partial_11
hess_inv[3,3] = 1/partial_22









