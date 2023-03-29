


# 5

# a

# Type 1 error is when we reject the null hypothesis when it is in fact true.
# Since the null hypothesis states that H_0: \theta = 0, we will simulate this 
# distribution, and see how often we reject this hypothesis, meaning that
# the data from N(0,1) suggests to us that the distribution is N(\theta, 1).

# The sequential testing algorithm: based on the description, I assume we need 
# to sample individual points in a loop, and if the test statistic is above c,
# we count that as a type 1 error occurence. There must be some limit to how
# many times we sample an individual point, say n times. So if the test
# statistic is above c at any point between 1 and n, type 1 error occured.

# This is the same as computing the test statistic for all n points, and
# then checking if any of them is above c. 

count = 0 # how many times we got type 1 error 
c = qnorm(0.975) # pre-specified critical value 1.96
nn = 1000 # number of MC runs

for (i in 1:nn)
{
  n = 1000
  z = rnorm(n, 0, 1)
  test = sqrt(seq(1,n)) * abs(cumsum(z)) / seq(1,n)
  
  if (max(test) > c) # type 1 error achieved if test stat > c
  {
    count = count + 1
  }
}
type1 = count / nn
type1

plot(seq(1,n), sqrt(seq(1,n)) * cumsum(z) / seq(1,n), type='l')
abline(h= c, col='red')
abline(h=-c, col='red')

# SEV NOTE: so this test graph exists, and prob(type 1) = 1.
# but choosing to be ignorant by not keeping track of this 
# statistic allows us to maintain the prob(type 1) at 5% ???
# it's like quantum mechanics, where the probability changes
# once we observe it. 




# b

do_test = function(c = 1.96)
{
  count = rep(0,5) 
  nn = 1000 # number of MC runs
  n = 5
  
  for (i in 1:nn)
  {
    z = rnorm(n, 0, 1)
    test = sqrt(seq(1,n)) * abs(cumsum(z)) / seq(1,n)
    
    hits = which(test > c) # indices of points > c
    if (length(hits) > 0) # type 1 error achieved if some point > c
    {
      idx = hits[1] # index of first element to be > c
      count[idx] = count[idx] + 1
    }
  }
  cat('p1..p5:', count, '\n')
  return(sum(count)/nn)
}

type1 = do_test(1.96)
type1

# is this type 1 probability what I was expecting? No it is not, 
# since we are expecting 5% by using the 1.96 quantile. 

type1 = do_test(2.13)
type1

type1 = do_test(2.4)
type1

# With this new threshold we get an error of 10%. But removing 
# the absolute value gives an error of 5%. This means that 
# 5% of the time, one of the five tests is above 2.13. 
# I believe this is an error in the question, because the 
# simulation with absolute value gives 5% when we set N=1, as expected. 
# A higher threshold is needed if we use absolute value,
# like something around 2.4. 
# It is assumed we are using a two-tailed 5% test since the test
# statistic includes an absolute value, and our quantile is 1.96. 



# c

# since sqrt(100)*.2 = 2 > 1.96, we have evidence against H_0 at 5% level.

# i)

count1 = 0 # how many times we got type 1 error 
count2 = 0
c = 1.96
nn = 10000 # number of MC runs

for (i in 1:nn)
{
  n = 100
  z = rnorm(n, 0, 1)
  test = sqrt(seq(1,n)) * abs(cumsum(z)) / seq(1,n)
  
  if (max(test) > c) # any test > c
  {
    count1 = count1 + 1
  }
  if (test[100] > c) # only 100th test > c
  {
    count2 = count2 + 1
  }
}
count1 / nn
count2 / nn

# running our test from part (a), we run 10_000 simulations, each sample 
# size 100. If we check that any of the 100 test statistics are above c,
# we get type 1 error rate of 37%. But if we only check the 100th test,
# the error rate is 5%. 

# Thus, the problem with my analysis for the scientist is that they may
# have stopped after seeing a good enough sample (since there is a 37%
# chance of that happening) instead of actually getting a result that 
# carries 5% significance. 

# If it is true that the scientist has pre-specified the number of 
# samples n=100 before running the experiment, then there is truly
# a 5% significance in their result. But if they modified n as they
# go along the experiment, then this result has 37% significance. 

# This is similar to throwing dice. The probability of getting a 3 is 1
# if you roll until you get a 3. Otherwise it is 1/6 if you roll once. 


# ii)

# this is a special case of part (b) where we are doing 2 sequential tests.
# Modifying the simulation and trying values, we see that c = 2.175 seems 
# to give about 5% type 1 error. 

# What is disturbing in this analysis is that we need to increase c 
# every time the scientist decides to re-run the experiment, after
# seeing that the previous run was not significant. From part (a) we know
# this statistic grows without bound so if the scientist really wanted 
# to get a significant result, they can just keep adding samples, and 
# they are guaranteed significance eventually.

do_test = function(c = 1.96)
{
  count = rep(0,2) 
  nn = 100000 # number of MC runs
  n = 2
  
  for (i in 1:nn)
  {
    z = rnorm(200, 0, 1)
    z1 = z[seq(1,100)]
    test1 = sqrt(100) * abs(mean(z1))
    test2 = sqrt(200) * abs(mean(z))
    
    if (test1 > c) 
    {
      count[1] = count[1] + 1
    }
    else 
    {
      if (test2 > c)
      {
        count[2] = count[2] + 1
      }
    }
  }
  cat('p1..p5:', count, '\n')
  return(sum(count)/nn)
}

type1 = do_test(2.175)
type1


# to calculate the p-value of this procedure having test < 2.1, 
# we can simulate it. 
# the conditions are that the first 100 samples are below c = 2.175,
# and then the second statistic is below 2.1.

do_test = function(c = 1.96)
{
  p = 0
  nn = 100000 # number of MC runs
  
  for (i in 1:nn)
  {
    z = rnorm(200, 0, 1)
    z1 = z[seq(1,100)]
    test1 = sqrt(100) * abs(mean(z1))
    test2 = sqrt(200) * abs(mean(z))
    
    if (test1 > c) 
    {
    }
    else 
    {
      if (test2 < 2.1)
      {
        p = p + 1
      }
    }
  }
  return(p/nn)
}
do_test(2.175)


# if the full data set of 200 points is published online, then a different
# scientist looking at it will use a critical value of 1.96 (relative
# to the standardized variable sqrt(200)xbar_200)
pnorm(2.1)
# The p-value this second scientist will see is 0.98 which is significant
# at the 5% level, even though in reality it is not due 
# to the sequential procedure. 

# This result does not agree with what the original scientist did in the 
# previous part because it doesn't take into account that they looked at 
# the data and decided to augment it after not getting significance at
# a sample size of 100. This makes the second scientist's test statistic
# invalid. 

1
# discuss whether we should use the same c if:
# 1. the scientist only takes 100 samples
# 2. the scientist says they discarded the first sample and took another batch
# basic question is should we still use the same c
# 100 samples x1..x100 ~ N(0,1)
# sqrt(100) xbar_100 > c reject H0: theta = 0
# sqrt(100) xbar_100 < c then generate x101... x200 ~ N(0,1)
# sqrt(200) > a


# part 1 says our porb of type 1 error is 100%
# this is the same as the graph from 830 
# where he showed it goes to 1

# basically c needs to change. thats it

# if we see 200 data points from the start, then its ok
# otherwise we are using information that its 

# n = 200
# sqrt(200) xbar_200 > 1.96 / c

