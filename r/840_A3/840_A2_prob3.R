




######################################################################
######################################################################
######################################################################
######################################################################


########################################################################
######################## T E S T Z O N E ###############################

# initialize population. fitness function is likelihood, want to maximize.
alphas = rep(NA,N) # using lists bc data frame keeps changing types
fitness = rep(NA,N)
strings = rep(NA,N)
for (i in 1:N)
{
  strings[i] = random_str()
  alphas[i] = str_to_dbl(strings[i])
  fitness[i] = lik(alphas[i], 0.1, data)
}
# the next generation
alphas2 = alphas
fitness2 = fitness
strings2 = strings

best_ever = list(NA,-Inf,'.') # keep track of best ever seen

get_best_ever = function(best_ever, alphas, fitness, strings)
{
  if (best_ever[[2]] > fitness[1])
    return(best_ever)
  else 
    return(list(alphas[1],fitness[1],strings[1]))
}


for (epoch in 1:10000)
{
  # sort by fitness
  sort_idx = rev(order(fitness))
  alphas = alphas[sort_idx]
  fitness = fitness[sort_idx]
  strings = strings[sort_idx]
  
  # update the best_ever
  best_ever = get_best_ever(best_ever, alphas, fitness, strings)
  
  # crossover: top N/2 will mate with each other (ignore duplicates)
  mate = matrix(nrow=2,ncol=N/2)
  mate[1,] = sample(seq(1,N/2),N/2,replace=T) #seq(1,N/2)
  mate[2,] = sample(seq(1,N),N/2,replace=T)
  
  for (i in 1:(N/2)) # for each mating pair
  {
    pair = mate[,i]
    p1 = pair[1] # the individual's index
    p2 = pair[2]
    s1 = strings[p1] # the individual's string
    s2 = strings[p2]
    
    # crossover: cut and paste genetic strings
    idx = sample(seq(1,64),1)
    n1 = paste(substring(s1,1,idx),substring(s2,idx+1),sep='')
    n2 = paste(substring(s2,1,idx),substring(s1,idx+1),sep='')
    
    # mutate 
    m1 = maybe_mutate(n1, mutate_p)
    m2 = maybe_mutate(n2, mutate_p)
    
    # update rows in the next generation: df2
    strings2[p1] = m1
    strings2[p2] = m2
    alphas2[p1] = str_to_dbl(m1)
    alphas2[p2] = str_to_dbl(m2)
    fitness2[p1] = lik(alphas2[p1],0.1, data)
    fitness2[p2] = lik(alphas2[p2],0.1, data)
  }
  # switch to next generation
  alphas = alphas2
  strings = strings2
  fitness = fitness2
}

best_ever
alphas

################ E N D T E S T Z O N E ###############################
######################################################################
######################################################################
######################################################################
























# a

data = c(-4.20, -2.85, -2.30, -1.02, 0.70, 0.98, 2.72, 3.50)

lik = function(a,b,data)
{
  n = length(data) 
  s = n*log(b) - n*log(pi)
  for (i in 1:n)
    s = s - log(b^2 + (data[i] - a)^2)
  return(s)
}

x = seq(-10,10,length=1000)
plot(x=x,y=lik(x,0.1, data),type='l')












# b

simulated_annealing = function(start)
{
  T0 = 10
  Tf = 1/10000000
  n = 1e6
  
  chain = rep(NA,n)
  chain[1] = start
  
  for (i in 2:n)
  {
    Ti = T0 * (Tf / T0)^((i-1)/n)
    old = chain[i-1]
    new = rnorm(1,old,1)
    
    if (runif(1) <= exp((lik(new,0.1,data) - lik(old,0.1,data)) / Ti))
      chain[i] = new
    else 
      chain[i] = old
  }
  return(chain)
}

par(mfrow=c(2,2))

for (start in c(-2.5, 0, -0.30875))
{
  c = simulated_annealing(start)
  c = c[1e5:1e6] # burn in 
  opt = c[length(c)] # optimal value
  
  plot(x=x,y=lik(x,0.1, data),type='l',main=paste('original, opt=',opt,sep=''))
  abline(v=opt, col="blue")
  plot(c, type='l',main='chain')
  hist(c,xlim=c(-10,10),main='histogram of chain')
  acf(c,main='chain acf')
}





















# c

# copy from part a
data = c(-4.20, -2.85, -2.30, -1.02, 0.70, 0.98, 2.72, 3.50)

lik = function(a,b,data)
{
  n = length(data) 
  s = n*log(b) - n*log(pi)
  for (i in 1:n)
    s = s - log(b^2 + (data[i] - a)^2)
  
  if (is.nan(s)) # for the sake of the fitness ordering, replace NaN w -Inf
    return(-Inf) 
  else
    return(s)
}

# a series of functions for dealing with strings representing 64 bits

empty_str = function() paste(rep('0',64),collapse='')

random_str = function() paste(sample(c("0","1"),64,replace=T), collapse='')

# https://gallery.rcpp.org/articles/strings_with_rcpp/
# https://stackoverflow.com/questions/50217954/double-precision-64-bit-representation-of-numeric-value-in-r-sign-exponent

Rcpp::cppFunction("
std::string dbl_to_str(double d, std::string s)
{
    uint64_t bits = *((uint64_t*)(&d));
    uint64_t bit;
    for (int i = 0; i < 64; i++)
    {
        bit = (bits >> i) & 0x1;
        s[i] = (char)(bit + 48);
    }
    return s;
}", plugins = "cpp11", includes = "#include <Rcpp.h>")

Rcpp::cppFunction("
double str_to_dbl(std::string s)
{
    uint64_t bits = 0;
    uint64_t bit; // must be uint64_t since it gets shifted 63 times
    for (int i = 0; i < 64; i++)
    {
        bit = s[i] - 48;
        bits |= (bit << i);
    }
    return *((double*)(&bits));
}", plugins = "cpp11", includes = "#include <Rcpp.h>")

# test the string to double function
str = empty_str() # allocate string on R side for gc
str = dbl_to_str(0.1, str)
str
str_to_dbl(str)

maybe_mutate = function(s, mutate_p) # mutate a string with given probability
{
  if (runif(1) <= mutate_p) # prob of mutation
  {
    idx = sample(seq(1,64),1) # index where to mutate
    if (substr(s,idx,idx)  == '0')
      substr(s,idx,idx) = '1'
    else 
      substr(s,idx,idx) = '0'
  }
  return(s)
}

update_row = function(str) # generate alpha and fitness given a string
{
  alpha = str_to_dbl(str)
  fitness = lik(alpha, 0.1, data)
  return(list(alpha, fitness, str))
}

get_best_ever = function(best_ever, df)
{
  if (best_ever[[2]] > df[1,2]) # compare best to this generations most fit
    return(best_ever)
  else 
    return(df[1,])
}

genetic = function(N = 10, mutate_p = 0.1, epochs = 1000)
{
  # initialize population. fitness function is likelihood, want to maximize.
  df = data.frame(matrix(nrow=N,ncol=3))
  colnames(df) = c("alpha","fitness","string")
  for (i in 1:N)
    df[i,] = update_row(random_str())
  
  # the next generation
  df2 = df
  
  # keep track of best ever seen
  best_ever = list(-Inf,-Inf,'.') 
  
  for (epoch in 1:epochs)
  {
    # sort by fitness
    df = df[order(-df$fitness), ]
    
    # update the best_ever
    best_ever = get_best_ever(best_ever, df)
    
    # crossover: pair top N/2 with the rest (ignore duplicates)
    mate = matrix(nrow=2,ncol=N/2)
    mate[1,] = sample(seq(1,N/2),N/2,replace=T)
    mate[2,] = sample(seq(1,N),N/2,replace=T)
    
    for (i in 1:(N/2)) # for each mating pair
    {
      pair = mate[,i]
      p1 = pair[1] # the individual's index
      p2 = pair[2]
      s1 = df$string[p1] # the individual's string
      s2 = df$string[p2]
      
      # crossover: cut and paste genetic strings
      idx = sample(seq(1,64),1)
      n1 = paste(substring(s1,1,idx),substring(s2,idx+1),sep='')
      n2 = paste(substring(s2,1,idx),substring(s1,idx+1),sep='')
      
      # mutate 
      m1 = maybe_mutate(n1, mutate_p)
      m2 = maybe_mutate(n2, mutate_p)
      
      # update rows in the next generation: df2
      df2[p1,] = update_row(m1)
      df2[p2,] = update_row(m2)
    }
    # switch to next generation
    df = df2
  }
  print(best_ever)
  return(df)
}


# the global optimum is about 0.73 according to part (b)
for (pop in c(10, 20, 30))
{
  for (mut in c(0.05, 0.1, 0.15, 0.3, 0.5))
  {
    print(paste("population: ", pop, " mutation: ", mut))
    df_ = genetic(pop, mut, 10000)
  }
}












