

# 2

# a 
# X ~ N(theta, 1)
# Xbar ~ N(theta, 1/10)
# variance of X is 1, variance of Xbar is 1/n, SD of Xbar is 1/sqrt(10)
thetas = seq(from = -1, to = 6, length.out = 100)
lik_a = dnorm(3.5, thetas, 1/sqrt(10))

# b
# 0 < Xbar < 5
# 0 - theta < Xbar - theta < 5 - theta
# sqrt(10)(-theta) < sqrt(10)(Xbar - theta) < sqrt(10)(5 - theta)
# know sqrt(10)(Xbar - theta) ~ N(0,1)
# = Phi{sqrt(10)(5 - theta)} - Phi{sqrt(10)(-theta)}
#   where Phi is the standard normal CDF
lik_b = pnorm(sqrt(10)*(5 - thetas)) - pnorm(-sqrt(10)*thetas)

# c
plot(thetas, lik_a, type='l', col='red')
lines(thetas, lik_b, col='green')
legend(x = 'topright', 
       legend=c('A', 'B'), 
       fill = c('red','green'))
lik_b
# comparison

# for part A, the likelihood is highest when theta = Xbar, which is 
# the usual MLE. It decreases around 3.5 in accordance to the 
# sample mean's variance, which is 1/sqrt(10) = 0.31

# for part B, we only know that the mean is in an interval. 
# standardizing this inequality gives us a probabbility statement.
# if theta was 0 then we have 50% chance the mean is to the right,
# and if theta is 5, 50% chance it's to the left.
# since SD = 0.31, 3 standard deviations = 1, we see that we quickly
# climb to 1 in the interval (0,5). this shows that if theta is here,
# then we are almost guaranteed to have Xbar in (0,5), except near 0,5. 

# naturally, having more precise information about Xbar makes the 
# likelihood that much more informative. 

