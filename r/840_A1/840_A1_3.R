

# 3
# X ~ Bin(5, theta)
# prior ~ Beta(a, b)

# using the beta posterior recursively, for each observation we end up
# with a posterior whose parameter A is equal to the prior's A, added
# the number of successes. The parameter B becomes the prior's B added
# the number of failures. So for the entire sample we end up adding 
# prior's A to the sum(x), 
# and B becomes (sample_size * 5) - sum(x) + prior B

# another way to think about this is that we have one experiment 
# from Bin(5*n, theta) since the sum of Binomials is Binomial. 

x = c(0, 1, 1, 2, 0, 0, 2, 3, 4, 3, 4, 1, 2, 5)

len_theta = 100
posterior = rep(0, len_theta)
thetas = seq(from = 0.2, to = 0.6, length.out = len_theta)

get_posterior = function(x, thetas, prior_a, prior_b, 
                         GET_MAP=FALSE, GET_BAYES=FALSE)
{
  n = length(x)
  post_a = prior_a + sum(x)
  post_b = prior_b + (n*5) - sum(x)
  
  if (GET_MAP) # MAP estimate
  {
    return((sum(x) + prior_a - 1) / ((n*5) + prior_a + prior_b - 2))
  }
  if (GET_BAYES) # Bayes estimate (mean of posterior)
  {
    return(post_a/(post_a + post_b))
  }
  return(dbeta(thetas, post_a, post_b))
}

plot(thetas, get_posterior(x, thetas, 100, 100), type='l', col='red')
lines(thetas, get_posterior(x, thetas, 10, 10), col='blue')
lines(thetas, get_posterior(x, thetas, 1, 1), col='green')
lines(thetas, get_posterior(x, thetas, 0.5,0.5), col='purple')
legend(x = 'topright', 
       legend=c('0.5', '1', '10', '100'), 
       fill = c('purple', 'green', 'blue', 'red'))

# Bayes estimates
b_0p5 = get_posterior(x, thetas, 0.5, 0.5, GET_BAYES = TRUE)
b_1   = get_posterior(x, thetas, 1, 1, GET_BAYES = TRUE)
b_10  = get_posterior(x, thetas, 10, 10, GET_BAYES = TRUE)
b_100 = get_posterior(x, thetas, 100, 100, GET_BAYES = TRUE)
c(b_0p5, b_1, b_10, b_100)
# [1] 0.4014085 0.4027778 0.4222222 0.4740741

# MAP estimates
map_0p5 = get_posterior(x, thetas, 0.5, 0.5, GET_MAP = TRUE)
map_1   = get_posterior(x, thetas, 1, 1, GET_MAP = TRUE)
map_10  = get_posterior(x, thetas, 10, 10, GET_MAP = TRUE)
map_100 = get_posterior(x, thetas, 100, 100, GET_MAP = TRUE)
c(map_0p5, map_1, map_10, map_100)
# [1] 0.6686747 0.6666667 0.6372549 0.5496454

