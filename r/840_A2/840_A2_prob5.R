

# a

P = matrix(ncol = 6, nrow = 6)
P[1,] = c( .5,  .5,   0,   0,   0,   0)
P[2,] = c(.25,  .5, .25,   0,   0,   0)
P[3,] = c(.25, .25, .25, .25,   0,   0)
P[4,] = c(  0,   0, .25, .25, .25, .25)
P[5,] = c(  0,   0,   0, .25,  .5, .25)
P[6,] = c(  0,   0,   0,   0,  .5,  .5)
Pt = t(P)

n = 1000
states = rep(NA, n)
states[1] = 1

for (i in 2:n)
{
  # https://stats.stackexchange.com/questions/67911/how-to-sample-from-a-discrete-distribution
  states[i] = sample(x = c(1,2,3,4,5,6), size = 1, replace = T, prob = P[states[i-1],])
}

par(mfrow=c(1,1))
plot(states, type='l', main='sample path')




# b

# compute relative frequency of simulation
freq = rep(NA,6)
for (i in 1:6)
{
  freq[i] = sum(states == i) / n
}

# guess value of stationary distribution pi
eig_ = eigen(Pt)

# the eigenvalue 1 is in first place
for (i in 1:6)
  print(eig_$values[i])

# its corresponding eigenvector
pi_ = matrix(nrow=6, ncol=1)
pi_[,1] = as.numeric(eig_$vectors[,1])
pi_ = pi_ / sum(pi_)

# compare relative freq to stationary dist
par(mfrow=c(1,2))
barplot(freq, 
        ylim=c(0,.3), 
        names.arg = c(1,2,3,4,5,6),
        main='relative frequency', 
        xlab='i')
barplot(pi_[,1], 
        ylim=c(0,.3), 
        names.arg = c(1,2,3,4,5,6), 
        main='stationary distribution',
        xlab='i')



# c

# compare equality up to numeric error
(pi_ - Pt %*% pi_) < 0.0000000000000001











# E X T R A 

# guess value of stationary distribution pi by observing dist at time n
pi = matrix(ncol = 1, nrow = 6)
pi[,1] = c(1,0,0,0,0,0) # initial distribution: start at state 1
pi_n = matrix(ncol = 6, nrow = n) # remember pi_n at each n
pi_n[1,] = pi
for (i in 2:n)
{
  pi_n[i,] = Pt %*% pi_n[i-1,]
}
pi = pi_n[n,]
# visualize how the probabilities change as time goes on
par(mfrow=c(1,1))
plot(pi_n[,1], ylim=c(0,0.6), type='l', col='red', main='state prob')
lines(pi_n[,2], ylim=c(0,0.6), type='l', col='green')
lines(pi_n[,3], ylim=c(0,0.6), type='l', col='blue')
lines(pi_n[,4], ylim=c(0,0.6), type='l', col='purple')
lines(pi_n[,5], ylim=c(0,0.6), type='l', col='cyan')
lines(pi_n[,6], ylim=c(0,0.6), type='l', col='orange')

