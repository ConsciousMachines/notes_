
# a

data_x = c(53,70,57,70,58,72,63,73,66,75,67,75,67,76,67,76,68,78,69,79,70,81,70)
data_y = c(2,1,1,1,1,0,1,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0)

p = function(a,b,x) 1 / (1 + exp(-a-b*x))

lik = function(a,b)
{
  l = 0
  for (i_ in 1:length(data_x))
  {
    xi = data_x[i_]
    yi = data_y[i_] # log(choose(6, yi)) +
    l = l + yi*log(p(a,b,xi)) + (6-yi)*log(1 - p(a,b,xi))
  }
  return(l)
}

partial_a = function(a,b)
{
  l = 0
  for (i_ in 1:length(data_x))
  {
    xi = data_x[i_]
    yi = data_y[i_]
    e_ = exp(-a-b*xi)
    l = l + (yi*e_ -6 +yi) / (1 + e_)
  }
  return(l)
}

partial_b = function(a,b)
{
  l = 0
  for (i_ in 1:length(data_x))
  {
    xi = data_x[i_]
    yi = data_y[i_]
    e_ = exp(-a-b*xi)
    l = l + (xi*(yi*e_ -6 +yi)) / (1 + e_)
  }
  return(l)
}

# we choose limits (-0.3, 0.3) for beta because larger causes overflow
# similarly, choose limits (-10, 10) for alpha 
N = 30
L = 0.1
x = seq(-L,L,length=N) * 100
y = seq(-L,L,length=N) * 3
z0 = outer(x,y,lik)
persp(x, y, z0, phi=0,theta=40, main='lik',xlab='alpha',ylab='beta')
z1 = outer(x,y,partial_a)
persp(x, y, z1, phi=0,theta=50, main='partial a',xlab='alpha',ylab='beta')
z2 = outer(x,y,partial_b)
persp(x, y, z2, phi=0,theta=40, main='partial b',xlab='alpha',ylab='beta')









# b

# for the optimization routine, make our likelihood negative to minimize.
lik_opt = function(x) -lik(x[1],x[2])

# try different starting point, in our grid 
N = 10 
L = 0.1
grid = matrix(nrow=N*N,ncol=3)
colnames(grid) = c('alpha','beta','lik')
xx = seq(-L,L,length=N) * 100
yy = seq(-L,L,length=N) * 3
for (i_ in 1:length(xx))
{
  for (j_ in 1:length(yy))
  {
    x_ = xx[i_]
    y_ = yy[j_]
    nm = optim(c(x_,y_), lik_opt, method="Nelder-Mead")
    grid[j_ + (i_-1)*N,] = c(x_, y_,-nm$value)
  }
}
grid

# for added clarity we can look at the 2D plots
plot(lik(nm$par[1],y),type='l', main='lik(alpha*,beta)',xlab='beta')
plot(lik(x, nm$par[2]),type='l',main='lik(alpha,beta*)',xlab='alpha')











# c

partial_a_opt = function(x) partial_a(x[1],x[2])
partial_b_opt = function(x) partial_b(x[1],x[2])
grad = function(x) c(partial_a_opt(x), partial_b_opt(x))

hess = function(pt)
{
  a = pt[1]
  b = pt[2]
  l1 = 0
  l2 = 0
  l3 = 0
  for (i_ in 1:length(data_x))
  {
    xi = data_x[i_]
    yi = data_y[i_]
    e_ = exp(-a-b*xi)
    p_aa = (-6*e_) / ((1 + e_)^2)
    p_ab = xi*p_aa
    p_bb = xi*p_ab
    l1 = l1 + p_aa
    l2 = l2 + p_ab
    l3 = l3 + p_bb
  }
  hess = matrix(data = c(l1,l2,l2,l3),nrow=2,ncol=2)
  return(hess)
}

hess_inv = function(pt)
{
  h = hess(pt)
  l1 = h[1,1]
  l2 = h[1,2]
  l3 = h[2,2]
  # inverse of 2x2 matrix
  det = l1*l3 - l2*l2
  hess = (1/det) * matrix(data = c(l3,-l2,-l2,l1),nrow=2,ncol=2)
  return(hess)
}

newton_step = function(old) # returns list of (point, flag)
{
  CALCULATION_IS_FINE = TRUE
  
  # grad
  g = matrix(grad(old), nrow=2,ncol=1)
  if (any(is.nan(g)) || any(is.infinite(g)))
  {
    #print("grad broken")
    CALCULATION_IS_FINE = FALSE
  }
  
  # solve system of equations
  if (FALSE)
  {
    # hessian
    h = hess(old)
    if (any(is.nan(h)) || any(is.infinite(h)))
    {
      #print("hess_inv broken")
      CALCULATION_IS_FINE = FALSE
    }
    
    ret = tryCatch(
      {
        list("point"=solve(h, -g) + old, "flag"=CALCULATION_IS_FINE)
      },
      error=function(cond) {
        message(cond)
        list("point"=matrix(nrow=2,ncol=1), "flag"=FALSE)
      },
      warning=function(cond) {
        message(cond)
        stop(cond)
      }
    ) 
    return(ret)
  }
  
  # matrix inverse method
  else 
  {
    # hessian
    h = hess_inv(old)
    if (any(is.nan(h)) || any(is.infinite(h)))
    {
      #print("hess_inv broken")
      CALCULATION_IS_FINE = FALSE
    }
    
    # new point
    new = old - h %*% g
    return(list("point"=new, "flag"=CALCULATION_IS_FINE))
  }
}

newton_raphson = function(old)
{
  step_ = newton_step(old)
  new   = step_$point
  flag  = step_$flag
  if (flag == FALSE) # nan/inf
    return(step_) 
  
  while (sum(abs(new-old)) > 0.001)
  {
    old   = new
    step_ = newton_step(old)
    new   = step_$point
    flag  = step_$flag
    if (flag == FALSE) # nan/inf
      break
  }
  return(step_)
}

# try different starting point, in our grid 
grid = matrix(nrow=N*N,ncol=3)
colnames(grid) = c('alpha','beta','Newton')
STOP_LOOPING = FALSE
for (i_ in 1:length(xx))
{
  if (STOP_LOOPING == TRUE) break
  
  for (j_ in 1:length(yy))
  {
    if (STOP_LOOPING == TRUE) break
    
    x_ = xx[i_]
    y_ = yy[j_]
    newt_ = newton_raphson(c(x_,y_))
    optimum = newt_$point
    flag = newt_$flag
    #if (flag == FALSE) # nan/inf in newton 

    lik_ = -lik_opt(optimum)
    grid[j_ + (i_-1)*N,] = c(x_, y_,lik_)
  }
}
grid












# e

# to feed gradient to optim, we add negative since its minimizing lik
grad_bfgs = function(x) -c(partial_a_opt(x), partial_b_opt(x))

grid = matrix(nrow=N*N,ncol=3)
colnames(grid) = c('alpha','beta','lik')
for (i_ in 1:length(xx))
{
  for (j_ in 1:length(yy))
  {
    x_ = xx[i_]
    y_ = yy[j_]
    o_ = optim(c(x_,y_), lik_opt, gr=grad_bfgs,method="BFGS")
    grid[j_ + (i_-1)*N,] = c(x_, y_,-o_$value)
  }
}
grid












