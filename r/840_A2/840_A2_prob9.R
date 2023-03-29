
# 1. regular MH (+ for Bayesians)
#          a = pi(y)q(y,x) / pi(x)q(x,y)

# 2. independent MH
#          g is independent? a = pi(y)g(x) / pi(x)g(y)

# 3. random walk MH
#          g is N lol    a = pi(y) / pi(x)

# 4. single component
#          a_i same as regular, just component-wise

# 5. Gibbs
#          CANNOT FIGURE OUT PI_i(Y_i) SO CANNOT BE DONE

# since our prior is Beta, we can pick a proposal distribution 
# with support = [0,1]. an "uninformative" option is Uniform(0,1).

# i dont really have a joint proposal distribution over the 6 variables,
# so an independent choice would be IND_MH or RANDOM_WALK_MH


log_post_theta_1 = function(p1,p2,p3)
{
  part1 = 375*log(p2*p1 + p3*(1-p1))
  part2 = 318*log(1 - p2*p1 - p3*(1-p1))
  part3 = 22*log(p1) + 15*log(1-p1)
  return(part1 + part2 + part3)
}

log_post_theta_2 = function(p1,p2,p3)
{
  part1 = 375*log(p2*p1 + p3*(1-p1))
  part2 = 318*log(1 - p2*p1 - p3*(1-p1))
  part3 = 17*log(p2) + 4*log(1-p2)
  return(part1 + part2 + part3)
}

log_post_theta_3 = function(p1,p2,p3)
{
  part1 = 375*log(p2*p1 + p3*(1-p1))
  part2 = 318*log(1 - p2*p1 - p3*(1-p1))
  part3 = 2*log(p3) + 12*log(1-p3)
  return(part1 + part2 + part3)
}

log_post_phi_1 = function(p1,p2,p3)
{
  part1 = 535*log(p2*p1 + p3*(1-p1))
  part2 = 701*log(1 - p2*p1 - p3*(1-p1))
  part3 = 31*log(p1) + 43*log(1-p1)
  return(part1 + part2 + part3)
}

log_post_phi_2 = function(p1,p2,p3)
{
  part1 = 535*log(p2*p1 + p3*(1-p1))
  part2 = 701*log(1 - p2*p1 - p3*(1-p1))
  part3 = 15*log(p2) + 15*log(1-p2)
  return(part1 + part2 + part3)
}

log_post_phi_3 = function(p1,p2,p3)
{
  part1 = 535*log(p2*p1 + p3*(1-p1))
  part2 = 701*log(1 - p2*p1 - p3*(1-p1))
  part3 = 10*log(p3) + 32*log(1-p3)
  return(part1 + part2 + part3)
}

####################################################################
####################################################################

log_accept_theta_1 = function(x, y, theta2, theta3)
{
  # log q = 0 since q = 1 for uniform density
  r1 = log_post_theta_1(y, theta2, theta3) - log_post_theta_1(x, theta2, theta3)
  return(min(r1, 0))
}

log_accept_theta_2 = function(x, y, theta1, theta3)
{
  r1 = log_post_theta_2(theta1, y, theta3) - log_post_theta_2(theta1, x, theta3)
  return(min(r1, 0))
}

log_accept_theta_3 = function(x, y, theta1, theta2)
{
  r1 = log_post_theta_3(theta1, theta2, y) - log_post_theta_3(theta1, theta2, x)
  return(min(r1, 0))
}

log_accept_phi_1 = function(x, y, phi2, phi3)
{
  r1 = log_post_phi_1(y, phi2, phi3) - log_post_phi_1(x, phi2, phi3)
  return(min(r1, 0))
}

log_accept_phi_2 = function(x, y, phi1, phi3)
{
  r1 = log_post_phi_2(phi1, y, phi3) - log_post_phi_2(phi1, x, phi3)
  return(min(r1, 0))
}

log_accept_phi_3 = function(x, y, phi1, phi2)
{
  r1 = log_post_phi_3(phi1, phi2, y) - log_post_phi_3(phi1, phi2, x)
  return(min(r1, 0))
}

# independent MH & component wise 

# using component wise because we need to optimize all parameters together, 
# we cannot do one at a time.
# in component wise, we only evaluate pi_i (which we have). in Gibbs, we 
# sample from it, which we cannot do.


# attempt 0 - seems to work, but i want to clean the code a bit
attempt_0_first = function(n = 10000)
{
  chain = matrix(ncol = 6, nrow = n)
  chain[1,] = 0.5
  
  for (i in 2:n)
  {
    for (j in 1:6)# iterate over components
    {
      # TODO: update these in the iteration
      # the parameters 
      theta1 = chain[i-1,1]
      theta2 = chain[i-1,2]
      theta3 = chain[i-1,3]
      phi1 = chain[i-1,4]
      phi2 = chain[i-1,5]
      phi3 = chain[i-1,6]
      
      x = chain[i-1,j]
      y = runif(1) # candidate
      logu = log(runif(1)) # compare to log acceptance ratio for stability
      
      # theta_1
      if (j == 1)
      {
        if (logu <= log_accept_theta_1(x, y, theta2, theta3))
          chain[i,j] = y
        else 
          chain[i,j] = x
      }
      
      # theta_2
      if (j == 2)
      {
        if (logu <= log_accept_theta_2(x, y, theta1, theta3))
          chain[i,j] = y
        else 
          chain[i,j] = x
      }
      
      # theta_3
      if (j == 3)
      {
        if (logu <= log_accept_theta_3(x, y, theta1, theta2))
          chain[i,j] = y
        else 
          chain[i,j] = x
      }
      
      # phi_1
      if (j == 4)
      {
        if (logu <= log_accept_phi_1(x, y, phi2, phi3))
          chain[i,j] = y
        else 
          chain[i,j] = x
      }
      
      # phi_2
      if (j == 5)
      {
        if (logu <= log_accept_phi_2(x, y, phi1, phi3))
          chain[i,j] = y
        else 
          chain[i,j] = x
      }
      
      # phi_3
      if (j == 6)
      {
        if (logu <= log_accept_phi_3(x, y, phi1, phi2))
          chain[i,j] = y
        else 
          chain[i,j] = x
      }
    }
  }
  return(chain)
}


# attempt 1 - best so far?
# cleaned up the iteration code a bit, making the updates
# to the parameters happen within the iteration.
# - - - 
# looking at the results, the histograms look smooth and are 
# consistent between runs.
# the plots of the chain also look fine but have a lot of flat regions.
# this means a lot of times we fail to accept a new point.
# This may be due to the choice of proposal distribution:
# choosing anything but uniform would imply prior knowledge of the 
# shape of the distribution, as Beta distributions vary wildly in shape.
# Wanting to remain neutral in this regard, of course the uniform 
# distribution will provide a lot of "misses" when we evaluate the 
# height of the density at the next step, since it jumps around uniformly.
# due to these long flat regions, we get very high autocorrelation in the 
# acf plot. 
attempt_1_better_code = function(n = 10000)
{
  chain = matrix(ncol = 6, nrow = n)
  chain[1,] = 0.5
  
  for (i in 2:n)
  {
    # the parameters, get updated with chain within the iteration
    theta1 = chain[i-1,1]
    theta2 = chain[i-1,2]
    theta3 = chain[i-1,3]
    phi1 = chain[i-1,4]
    phi2 = chain[i-1,5]
    phi3 = chain[i-1,6]
    
    x = chain[i-1,]
    y = runif(6) # candidate
    logu = log(runif(6)) # compare to log acceptance ratio for stability
    
    # theta_1
    if (logu[1] <= log_accept_theta_1(x[1], y[1], theta2, theta3))
    {
      chain[i,1] = y[1]
      theta1 = y[1]
    }
    else 
      chain[i,1] = x[1]
    
    # theta_2
    if (logu[2] <= log_accept_theta_2(x[2], y[2], theta1, theta3))
    {
      chain[i,2] = y[2]
      theta2 = y[2]
    }
    else 
      chain[i,2] = x[2]
    
    # theta_3
    if (logu[3] <= log_accept_theta_3(x[3], y[3], theta1, theta2))
    {
      chain[i,3] = y[3]
      theta3 = y[3]
    }
    else 
      chain[i,3] = x[3]
    
    # phi_1
    if (logu[4] <= log_accept_phi_1(x[4], y[4], phi2, phi3))
    {
      chain[i,4] = y[4]
      phi1 = y[4]
    }
    else 
      chain[i,4] = x[4]
    
    # phi_2
    if (logu[5] <= log_accept_phi_2(x[5], y[5], phi1, phi3))
    {
      chain[i,5] = y[5]
      phi2 = y[5]
    }
    else 
      chain[i,5] = x[5]
    
    # phi_3
    if (logu[6] <= log_accept_phi_3(x[6], y[6], phi1, phi2))
    {
      chain[i,6] = y[6]
      phi3 = y[6]
    }
    else 
      chain[i,6] = x[6]
  }
  return(chain)
}


# attempt 2 - meh
# i changed 1 line of code to make Y come from the prior rather than 
# from uniform. there is relatively less autocorrelation but flat
# regions still exist. 
attempt_2_prior_as_proposal = function(n = 10000)
{
  log_accept_theta_1 = function(x, y, theta2, theta3)
  {
    r1 = log_post_theta_1(y, theta2, theta3) - log_post_theta_1(x, theta2, theta3)
    r1 = r1 + log(dbeta(x, 23, 16)) - log(dbeta(y, 23,16))
    return(min(r1, 0))
  }
  
  log_accept_theta_2 = function(x, y, theta1, theta3)
  {
    r1 = log_post_theta_2(theta1, y, theta3) - log_post_theta_2(theta1, x, theta3)
    r1 = r1 + log(dbeta(x, 18,5)) - log(dbeta(y, 18,5))
    return(min(r1, 0))
  }
  
  log_accept_theta_3 = function(x, y, theta1, theta2)
  {
    r1 = log_post_theta_3(theta1, theta2, y) - log_post_theta_3(theta1, theta2, x)
    r1 = r1 + log(dbeta(x, 3,13)) - log(dbeta(y, 3,13))
    return(min(r1, 0))
  }
  
  log_accept_phi_1 = function(x, y, phi2, phi3)
  {
    r1 = log_post_phi_1(y, phi2, phi3) - log_post_phi_1(x, phi2, phi3)
    r1 = r1 + log(dbeta(x, 32,44)) - log(dbeta(y, 32,44))
    return(min(r1, 0))
  }
  
  log_accept_phi_2 = function(x, y, phi1, phi3)
  {
    r1 = log_post_phi_2(phi1, y, phi3) - log_post_phi_2(phi1, x, phi3)
    r1 = r1 + log(dbeta(x, 16,16)) - log(dbeta(y, 16,16))
    return(min(r1, 0))
  }
  
  log_accept_phi_3 = function(x, y, phi1, phi2)
  {
    r1 = log_post_phi_3(phi1, phi2, y) - log_post_phi_3(phi1, phi2, x)
    r1 = r1 + log(dbeta(x, 11,33)) - log(dbeta(y, 11,33))
    return(min(r1, 0))
  }
  
  chain = matrix(ncol = 6, nrow = n)
  chain[1,] = 0.5
  
  for (i in 2:n)
  {
    # the parameters, get updated with chain within the iteration
    theta1 = chain[i-1,1]
    theta2 = chain[i-1,2]
    theta3 = chain[i-1,3]
    phi1 = chain[i-1,4]
    phi2 = chain[i-1,5]
    phi3 = chain[i-1,6]
    
    x = chain[i-1,]
    y1 = rbeta(1, 23,16)
    y2 = rbeta(1, 18,5)
    y3 = rbeta(1, 3,13)
    y4 = rbeta(1, 32,44)
    y5 = rbeta(1, 16,16)
    y6 = rbeta(1, 11,33)
    y = c(y1,y2,y3,y4,y5,y6) # candidate
    logu = log(runif(6)) # compare to log acceptance ratio for stability
    
    # theta_1
    if (logu[1] <= log_accept_theta_1(x[1], y[1], theta2, theta3))
    {
      chain[i,1] = y[1]
      theta1 = y[1]
    }
    else 
      chain[i,1] = x[1]
    
    # theta_2
    if (logu[2] <= log_accept_theta_2(x[2], y[2], theta1, theta3))
    {
      chain[i,2] = y[2]
      theta2 = y[2]
    }
    else 
      chain[i,2] = x[2]
    
    # theta_3
    if (logu[3] <= log_accept_theta_3(x[3], y[3], theta1, theta2))
    {
      chain[i,3] = y[3]
      theta3 = y[3]
    }
    else 
      chain[i,3] = x[3]
    
    # phi_1
    if (logu[4] <= log_accept_phi_1(x[4], y[4], phi2, phi3))
    {
      chain[i,4] = y[4]
      phi1 = y[4]
    }
    else 
      chain[i,4] = x[4]
    
    # phi_2
    if (logu[5] <= log_accept_phi_2(x[5], y[5], phi1, phi3))
    {
      chain[i,5] = y[5]
      phi2 = y[5]
    }
    else 
      chain[i,5] = x[5]
    
    # phi_3
    if (logu[6] <= log_accept_phi_3(x[6], y[6], phi1, phi2))
    {
      chain[i,6] = y[6]
      phi3 = y[6]
    }
    else 
      chain[i,6] = x[6]
  }
  return(chain)
}


# similar results to the others although for sd=0.05 we get less
# autocorrelation, although still too much.
attempt_3_random_walk = function(n = 10000)
{
  chain = matrix(ncol = 6, nrow = n)
  chain[1,] = 0.5
  
  for (i in 2:n)
  {
    # the parameters, get updated with chain within the iteration
    theta1 = chain[i-1,1]
    theta2 = chain[i-1,2]
    theta3 = chain[i-1,3]
    phi1 = chain[i-1,4]
    phi2 = chain[i-1,5]
    phi3 = chain[i-1,6]
    
    x = chain[i-1,]
    e = rnorm(6, mean=0, sd=0.05)
    y = pmax(pmin(x + e, 1), 0) # candidate
    logu = log(runif(6)) # compare to log acceptance ratio for stability
    
    # theta_1
    if (logu[1] <= log_accept_theta_1(x[1], y[1], theta2, theta3))
    {
      chain[i,1] = y[1]
      theta1 = y[1]
    }
    else 
      chain[i,1] = x[1]
    
    # theta_2
    if (logu[2] <= log_accept_theta_2(x[2], y[2], theta1, theta3))
    {
      chain[i,2] = y[2]
      theta2 = y[2]
    }
    else 
      chain[i,2] = x[2]
    
    # theta_3
    if (logu[3] <= log_accept_theta_3(x[3], y[3], theta1, theta2))
    {
      chain[i,3] = y[3]
      theta3 = y[3]
    }
    else 
      chain[i,3] = x[3]
    
    # phi_1
    if (logu[4] <= log_accept_phi_1(x[4], y[4], phi2, phi3))
    {
      chain[i,4] = y[4]
      phi1 = y[4]
    }
    else 
      chain[i,4] = x[4]
    
    # phi_2
    if (logu[5] <= log_accept_phi_2(x[5], y[5], phi1, phi3))
    {
      chain[i,5] = y[5]
      phi2 = y[5]
    }
    else 
      chain[i,5] = x[5]
    
    # phi_3
    if (logu[6] <= log_accept_phi_3(x[6], y[6], phi1, phi2))
    {
      chain[i,6] = y[6]
      phi3 = y[6]
    }
    else 
      chain[i,6] = x[6]
  }
  return(chain)
}

# TODO: make a joint distribution?





chain = attempt_3_random_walk(n = 10000)
par(mfrow=c(3,2))
for (i in 1:6) acf(chain[,i])
for (i in 1:6) plot(chain[,i], type='l')
for (i in 1:6) hist(chain[,i])

get_SE = function(x) # return variance of sample mean
{
  n = length(x)
  mu = sum(x) / n
  s_sq = sum((x - mu)^2) / ((n-1)*n) # notes 2 p 30
  return(sqrt(s_sq))
}

# point estimates of the 6 parameters
thetas = matrix(nrow=1,ncol=6)
thetas[1,] = colSums(chain) / length(chain[,1])
colnames(thetas) = c('theta_1', 'theta_2', 'theta_3', 'phi_1', 'phi_2','phi_3')
round(thetas, 4)

stderrs = matrix(nrow=1,ncol=6)
stderrs[1,] = c(get_SE(chain[,1]), get_SE(chain[,2]), get_SE(chain[,3]), get_SE(chain[,4]), get_SE(chain[,5]), get_SE(chain[,6]))
colnames(stderrs) = c('SE_theta_1', 'SE_theta_2', 'SE_theta_3', 'SE_phi_1', 'SE_phi_2','SE_phi_3')
round(stderrs, 6)





# since we do not know the distribution of this statistic, we will simulate it
# and get the quantiles as the credible interval. 

OR = function(t1, p1)
{
  return((t1 * (1-p1)) / (p1*(1-t1)))
}

chain = attempt_3_random_walk(n = 1000000)
chain = chain[1000:1000000,] # burn-in
ORs = OR(chain[,1], chain[,4])

# 95% credible interval of Odds Ratio
quantile(ORs, probs = c(0.025, 0.975)) 
# point estimate of Odds Ratio
mean(ORs)

par(mfrow=c(1,1))
hist(ORs)
