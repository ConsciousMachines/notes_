
log_post_pi = function(alp, eta, b,c,x)
{
  n = length(x)
  p1 = (-alp -c*eta)
  p2 = (n + b - 1)*log(eta)
  p3 = n * log(alp)
  p4 = (alp-1)*log(x) -eta * (x^alp)
  return(p1 + p2 + p3 + sum(p4))
}

post_pi = function(alp, eta, b,c,x)
{
  n = length(x)
  p1 = exp(-alp -c*eta)
  p2 = eta^(n + b - 1)
  p3 = alp^n
  p4 = (x^(alp-1)) * exp(-eta * (x^alp))
  return(p1 * p2 * p3 * prod(p4))
}

log_p = function(a2,n2,a1,n1, b,c,x)
{
  n = length(x)
  p1 = -a2/a1 - n2/n1 + a1/a2 + n1/n2 + a1 + c*n1 -a2 -c*n2
  p2 = (n+b)*log(n2) + (n+1)*log(a2) - (n+b)*log(n1) - (n+1)*log(a1)
  p3 = sum((a2-a1)*log(x) + ((n1*(x^a1)) - (n2*(x^a2))))
  return(min(p1+p2+p3,0))
}

p = function(a2,n2,a1,n1, b,c,x)
{
  n = length(x)
  p1 = exp(-a2/a1 - n2/n1 + a1/a2 + n1/n2 + a1 + c*n1 -a2 -c*n2)
  p2 = ((n2^(n+b)) * (a2^(n+1))) / ((n1^(n+b)) * (a1^(n+1)))
  p3 = prod(((x^(a2-a1)) * exp(((n1*(x^a1)) - (n2*(x^a2))))))
  return(min(p1*p2*p3, 1))
}


# c



run = function(b,c,x, NN = 50000, plot_graf=F)
{
  chain = matrix(nrow=NN, ncol=2)
  chain[1,] = c(1,1)
  
  for (i in 2:NN)
  {
    a1 = chain[i-1,1]
    n1 = chain[i-1,2]
    
    a2 = rexp(1, 1/a1)
    n2 = rexp(1, 1/n1)
    
    if (log(runif(1)) <= log_p(a2,n2,a1,n1, b,c,x))
      chain[i,] = c(a2,n2)
    else 
      chain[i,] = chain[i-1,]
  }
  
  if (plot_graf) # plot graphs
  {
    par(mfrow = c(3,2))
    hist(chain[,1], main=paste0("(b=", b, " :: c=", c, ")"))
    hist(chain[,2])
    plot(chain[,1],type='l')
    plot(chain[,2],type='l')
    acf(chain[,1])
    acf(chain[,2])
  }
  
  return(chain)
}


# d

# posterior mean
b = 5
c = 5
x = c(0.56, 2.26, 1.90, 0.94, 1.40, 1.39, 1.00, 1.45, 2.32, 2.08, 0.89, 1.68)
NN = 50000
chain = run(b,c,x,NN,T)
c(alpha=mean(chain[,1]),eta=mean(chain[,2]))


# e

# MAP estimate
MAPs = rep(NA, NN)
for (i in 1:NN) MAPs[i] = post_pi(chain[i,1],chain[i,2],b,c,x)
map_idx = which(MAPs == max(MAPs))[1]
c(chain[map_idx,1],chain[map_idx,2])

# 95% credible interval
quantile(chain[,1], probs = c(0.025, 0.975))
quantile(chain[,2], probs = c(0.025, 0.975))














par(mfrow = c(3,2))
hist(chain[,1])
hist(chain[,2])
plot(chain[,1],type='l')
plot(chain[,2],type='l')
acf(chain[,1])
acf(chain[,2])










MM = 15
params = rep(1, MM)
for (i in 2:MM) params[i] = 1.4*params[i-1]
params
for (i in params)
{
  for (j in params)
  {
    tryCatch({
      run(b=i, c=j, x, NN=10000, T)
    }, warning = function(w) {
      print(paste(i,j,e))
    }, error = function(e) {
      print(paste(i,j,e))
    }, finally = {
    })
  }
}




