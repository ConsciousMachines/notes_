




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















poisson_mixture_em <- function(data, p_init, lambda1_init, lambda2_init, tol = 1e-8, max_iter = 1000) {
  n <- length(data)
  p <- p_init
  lambda1 <- lambda1_init
  lambda2 <- lambda2_init
  log_likelihood <- -Inf
  iter <- 0
  
  while (iter < max_iter) {
    iter <- iter + 1
    
    # E-step
    poisson1_prob <- p * dpois(data, lambda1)
    poisson2_prob <- (1 - p) * dpois(data, lambda2)
    denom <- poisson1_prob + poisson2_prob
    resp1 <- poisson1_prob / denom
    resp2 <- poisson2_prob / denom
    
    # M-step
    p_new <- mean(resp1)
    lambda1_new <- sum(resp1 * data) / sum(resp1)
    lambda2_new <- sum(resp2 * data) / sum(resp2)
    
    # Convergence check
    log_likelihood_new <- sum(log(poisson1_prob + poisson2_prob))
    if (abs(log_likelihood_new - log_likelihood) < tol) {
      break
    }
    
    p <- p_new
    lambda1 <- lambda1_new
    lambda2 <- lambda2_new
    log_likelihood <- log_likelihood_new
  }
  
  return(list(
    p = p,
    lambda1 = lambda1,
    lambda2 = lambda2,
    log_likelihood = log_likelihood,
    iterations = iter
  ))
}

data <- c(24, 18, 21, 5, 5, 11, 11, 17, 6, 7, 20, 13, 4, 16, 19, 21, 4, 22, 8, 17)
p_init <- 0.3
lambda1_init <- 5
lambda2_init <- 15
result <- poisson_mixture_em(data, p_init, lambda1_init, lambda2_init)

cat("Estimated p:", result$p, "\n")
cat("Estimated lambda1:", result$lambda1, "\n")
cat("Estimated lambda2:", result$lambda2, "\n")
cat("Log-likelihood:", result$log_likelihood, "\n")
cat("Iterations:", result$iterations, "\n")
    