

$$
  \begin{aligned}
U &\leq \min \biggl\{ \frac{\pi(\alpha^*, \eta^* \mid \mathbb{X}) q(\alpha^*, \eta^* \mid \alpha_{(t)}, \eta_{(t)})}{\pi(\alpha_{(t)}, \eta_{(t)} \mid \mathbb{X}) q(\alpha_{(t)}, \eta_{(t)} \mid \alpha^*, \eta^*)}, 1 \biggr\} \\
\log U &\leq \log \min \biggl\{ \frac{\pi(\alpha^*, \eta^* \mid \mathbb{X}) q(\alpha^*, \eta^* \mid \alpha_{(t)}, \eta_{(t)})}{\pi(\alpha_{(t)}, \eta_{(t)} \mid \mathbb{X}) q(\alpha_{(t)}, \eta_{(t)} \mid \alpha^*, \eta^*)}, 1 \biggr\} \\
\log U &\leq \min \biggl\{ \log \frac{\pi(\alpha^*, \eta^* \mid \mathbb{X}) q(\alpha^*, \eta^* \mid \alpha_{(t)}, \eta_{(t)})}{\pi(\alpha_{(t)}, \eta_{(t)} \mid \mathbb{X}) q(\alpha_{(t)}, \eta_{(t)} \mid \alpha^*, \eta^*)}, 0 \biggr\} \\
\log \frac{\pi(\alpha^*, \eta^* \mid \mathbb{X}) q(\alpha^*, \eta^* \mid \alpha_{(t)}, \eta_{(t)})}{\pi(\alpha_{(t)}, \eta_{(t)} \mid \mathbb{X}) q(\alpha_{(t)}, \eta_{(t)} \mid \alpha^*, \eta^*)} &= \log \pi(\alpha^*, \eta^* \mid \mathbb{X}) + \log q(\alpha^*, \eta^* \mid \alpha_{(t)}, \eta_{(t)}) \\ 
& \qquad - \log \pi(\alpha_{(t)}, \eta_{(t)} \mid \mathbb{X}) - \log q(\alpha_{(t)}, \eta_{(t)} \mid \alpha^*, \eta^*) \\
\end{aligned}
$$

#   M E T R O P O L I S 

# we do a random walk on teh x axis
# if next step has higher density graph, go there
# if next step has lower graph, go there WP f(j)/f(i), else stay

n = 5000

lam = 10 # lambda param of poisson

chain = rep(NA,n)
chain[1] = 0 # initial state 

for (idx in 2:n)
{
  i = chain[idx-1] # previous state
  
  # generate j, random walk
  if (i == 0)
    j = sample(x = c(0,1), size = 1, prob = c(.5, .5))
  else
    j = sample(x = c(i-1,i+1), size = 1, prob = c(.5, .5))
  
  # set r 
  r = dpois(j, lam) / dpois(i, lam)
  
  # set next step
  if (r >= 1) 
    chain[idx] = j
  else 
  {
    if (runif(1) < r) 
      chain[idx] = j
    else 
      chain[idx] = chain[idx-1]
  }
}

par(mfrow=c(2,2))
hist(chain[1000:n], xlim=c(0,25))
hist(rpois(n-1000, lam), xlim=c(0,25))
plot(chain[1000:n], type = 'l')




#   M E T R O P O L I S - H A S T I N G S

# 1. start at state i
# 2. generate j from prob dist q_ij
#      starting at x, next state is generated from q(x,y) * * * 
# 3. r = pi_j q_ji / pi_i q_ij 
# 4. if r > 1 ... same as metro

# see the steps just above Example 4
# exampel 4 - in the candidate dist, x is the mean

# q(x,y) is proposal generating density, when process is at x, 
#   a value y is generated from this density.

lam = 1.5
x_ = seq(0,5,1/10)
y_ = dexp(x_, lam)
plot(x=x_, y=y_, type='l', ylim=c(0,1.5))







# c

p = function(as, at, ns, nt, c, b, n, x)
{
  p1 = exp(-as/at -ns/nt +at/as + nt/ns +at +c*nt -as -c*ns)
  p2 = (ns/nt)^(n+b) * (as/at)^(n+1)
  p3 = prod(x^(as-at) * exp(nt*x^at - ns * x^as))
  return(min(p1*p2*p3, 1))
}

run = function(x,len,c,b)
{
  NN = 1000
  alp_n = rep(NA,NN)
  eta_n = rep(NA,NN)
  
  # step 1
  alp_n[1] = 1 # initial alpha
  eta_n[1] = 1 # initial eta 
  
  for (i in 2:NN)
  {
    alpha = alp_n[i-1]
    eta = eta_n[i-1]
    
    # step 2
    alpha_star = rexp(1, 1/alpha)
    eta_star = rexp(1, 1/eta)
    u = runif(1)
    
    # step 3
    if (u <= p(alpha_star, alpha, eta_star, eta, c, b, len, x))
    {
      alp_n[i] = alpha_star
      eta_n[i] = eta_star
    } else 
    {
      alp_n[i] = alpha
      eta_n[i] = eta
    }
  }
  
  par(mfrow=c(3,2))
  hist(alp_n)
  hist(eta_n)
  plot(alp_n, type='l')
  plot(eta_n, type='l')
  acf(alp_n)
  acf(eta_n)
}









NN = 1000
alp_n = rep(NA,NN)
eta_n = rep(NA,NN)

# step 1
alp_n[1] = 1 # initial alpha
eta_n[1] = 1 # initial eta 

for (i in 2:NN)
{
  alpha = alp_n[i-1]
  eta = eta_n[i-1]
  
  # step 2
  alpha_star = rexp(1, 1/alpha)
  eta_star = rexp(1, 1/eta)
  u = runif(1)
  
  # step 3
  if (u <= p(alpha_star, alpha, eta_star, eta, c, b, len, x))
  {
    alp_n[i] = alpha_star
    eta_n[i] = eta_star
  } else 
  {
    alp_n[i] = alpha
    eta_n[i] = eta
  }
}

mean(alp_n)

mean(eta_n)

quantile(alp_n, probs = c(0.025, 0.975)) 

quantile(eta_n, probs = c(0.025, 0.975)) 


post_pi = function(alp, eta)
{
  n = length(x)
  p1 = exp(-alp - c*eta)
  p2 = eta^(n+b-1)
  p3 = alp^n
  p4 = x^(alp-1) * exp(-eta * x^alp)
  return(p1 * p2 * p3 * prod(p4))
}

max(post_pi(alp_n))
1













# hyper params
x = c(0.56, 2.26, 1.90, 0.94, 1.40, 1.39, 1.00, 1.45, 2.32, 2.08, 0.89, 1.68)
len = length(x)

run(x,len,c=0.1,b=1)
run(x,len,c=4,b=1)
run(x,len,c=4,b=0.1)
run(x,len,c=10,b=0.1)
run(x,len,c=0.1,b=0.1)
run(x,len,c=0.1,b=4)
run(x,len,c=0.1,b=10)

max(post_pi(eta_n))
1




log_post_pi = function(alp, eta, x, c,b)
{
  n = length(x)
  p1 = (-alp - c*eta)
  p2 = (n+b-1)*log(eta)
  p3 = n*log(alp)
  p4 = (alp-1)*log(x) + (-eta * x^alp)
  return(p1 + p2 + p3 + sum(p4))
}

log_q = function(a2,n2,a1,n1)
{
  p1 = -(log(a1) + log(n1))
  p2 = (-a2/a1 -n2/n1)
  return(p1 + p2)
}

log_accept = function(a2,n2,a1,n1)
{
  r1 = log_post_pi(a2,n2) - log_post_pi(a1,n1) + log_q(a2,n2,a1,n1) - log_q(a1,n1,a2,n2)
  return(min(r1, 0))
}




#######################################################################

# globals
x = c(0.56, 2.26, 1.90, 0.94, 1.40, 1.39, 1.00, 1.45, 2.32, 2.08, 0.89, 1.68)
b = 1
c = 1


post_pi = function(alp, eta)
{
  n = length(x)
  p1 = exp(-alp - c*eta)
  p2 = eta^(n+b-1)
  p3 = alp^n
  p4 = x^(alp-1) * exp(-eta * x^alp)
  return(p1 * p2 * p3 * prod(p4))
}

q = function(a2,n2,a1,n1)
{
  p1 = 1 / (a1 * n1)
  p2 = exp(-a2/a1 -n2/n1)
  return(p1*p2)
}

accept = function(a2,n2,a1,n1)
{
  r1 = post_pi(a2,n2) / post_pi(a1,n1)
  r2 = q(a2,n2,a1,n1) / q(a1,n1,a2,n2)
  return(min(r1*r2, 1))
}

NN = 10000
chain = matrix(nrow=NN, ncol=2)
chain[1,] = c(2,2)

for (i in 2:NN)
{
  a1 = chain[i-1,1]
  n1 = chain[i-1,2]
  
  # step 2
  a2 = rexp(1, 1/a1)
  n2 = rexp(1, 1/n1)

  # step 3
  if (runif(1) <= accept(a2,n2,a1,n1))
    chain[i,] = c(a2,n2)
  else 
    chain[i,] = chain[i-1,]
}
accept(a2,n2,a1,n1)


par(mfrow=c(3,2))
hist(chain[,1])
hist(chain[,2])
plot(chain[,1], type='l')
plot(chain[,2], type='l')
acf(chain[,1])
acf(chain[,2])

chain[,1]

