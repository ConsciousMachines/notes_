# 1
n = 100
results1 = rep(0, n)
results2 = rep(0, n)
theta = seq(from = 0, to = 10, length.out = n)

for (i in 1:n)
{
  results1[i] = dgamma(2, 3, theta[i])
  results2[i] = dpois(3, theta[i]*2)
}

plot(theta, results1, type='l', col='red')
lines(theta, results2, col='green')
legend(x = 'topright', 
       legend=c('A gam', 'B pois'), 
       fill = c('red','green'))

# part 3
# comparing the likelihoods, we see that L1 is proportional 
# to L2 as a function of theta. The MLE will thus be the same,
# yielding theta* = 3/2. 
# discussion:
# according to the Strong Likelihood Principle, 
# the likelihood fn contains all info x has about theta.
# if x, y are two observations (possibly from different models)
# satisfying L1 = c L2, for every theta, they carry the same
# information about theta and must lead to identical inference.
# the constraints are satisfied in this situation, meaning
# both experimenters A and B have the same inference about 
# theta, given their chosen models and the observations X,Y.
# their likelihood functions are a proportional. 
# so A and B make identical inference about theta.

