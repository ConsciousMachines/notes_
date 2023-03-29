# Neural Networks From First Principles

For people who like to understand how things work by building a small version of it from scratch, it has never been a better time to learn (make) neural networks. This is my journey to implement NNs from first principles, one neuron at a time. 

# Step 1: XOR
All the math and under the hood knowledge you will need is in chapter 1 of Nielsen's awesome blog: (don't read the code! you will make it yourself)

http://neuralnetworksanddeeplearning.com/chap1.html

And you need this to understand the connection between derivatives and graphs:

http://colah.github.io/posts/2015-08-Backprop/

Now after reading this, you are ready code up your own network using this example:

https://iamtrask.github.io/2015/07/12/basic-python-network/

This article by Trask inspired me back in the day. There is a lot of "intuition" here that may be overwhelming. Forget all that, and just take the mathematical equation of the network, which is:
```
cost = (output - y)^2
output = sigmoid(h1 * w31 + h2 * w32 + b3)
h1 = sigmoid(x1 * x11 + x2 * x12 + b1)
h2 = sigmoid(x1 * x21 + x2 * x22 + b3)
```
and get the partial derivatives. The intuition will come after you derive these equations yourself. Additionally, Trask wrote a book "Grokking Deep Learning" that I am about to read, full of intuition that I suggest is best left after you derive some more equations yourself. Once you program an initial neural net, compare your answer with this article, which has a nice batched matrix form of this network:

https://towardsdatascience.com/how-neural-networks-solve-the-xor-problem-59763136bdd7

# Step 2: MNIST
After getting XOR to work, making MNIST work is only a question of expanding input size from 2 to 784, hidden unit size from 2 to 30, and output size from 1 to 10. Do this all on paper, only to realize that having an output size of 10 is equivalent to a stack of ten nets with an output of 1. You might need some matrix derivatives, which can be found here:

https://mostafa-samir.github.io/auto-diff-pt2/

If you choose to do batching, you will encounter some wonky matrix algebra. The derivatives are best written with respect to Y, where Y is the pre-activation:
```
Y = W * H_i-1 + B
H_i = sigmoid(Y)
```
One divine level skill to learn is using np.einsum:

https://ajcr.net/Basic-guide-to-einsum/

https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/

When I did this, I did peek at Nielsen's code. I did not understand a single thing that was going on. That is why I did everything myself from scratch. One cool thing I did notice though is that his network architecture is written like this:
```
network_structure = [num_input, 30, 30, num_output]
```
Using this as a guide to how you structure your Jacobian multiplications to count the paths (chain rule derivatives) from node to node, you can come up with a generalized fully connected network that can be expanded to any depth and width! 
After that is done, it would be a good time to compare your code with Nielsen's. I thought there would be some crazy matrix algebra to do batching, but it's just a for loop over the samples. 
Now with working MNIST, it's time for some fun:

https://ml.berkeley.edu/blog/posts/adversarial-examples/

Create your own adversarial examples by modifying the cost function. You already have everything you need for this, just restructure the training loop (derive the cost gradient with respect to X on paper)

# Step 2.1: Adversarial MNIST

I had a hypothesis that since adversarial examples look like noise, maybe if we add noise to the training data, the network will learn to ignore noise and look for "higher level" features. Well it seems I was right! (not sure if a paper exists on this). Since I did this part in Jax, I spent most of the time tuning parameters to avoid NaNs, in the mean-time implementing batch-normalization, weight regularization, and screwing around with hyper parameters until I found a magical combination that doesn't return NaNs. 
```
x += jax.random.normal(mk, shape = x.shape) * 0.1 + 0.3 # add Gaussian noise to each batch
```
First I trained the normal network. Then I used this network to create 50_000 adversarial examples. Checking them, it got 1_515 / 50_000 correct! Then I trained a network with the above line included, it was a crappy one with about 90% accuracy, but testing it on those same adversarial examples, it got 45_903 / 50_000 correct. Wahoo. 

# Step 3: CNNs

*under construction*

# Step 4: JAX

After going through all that, you should feel pretty confident in what is going on under the hood. We are not going to take that away! Most people at this stage move on to Tensorflow or PyTorch. Instead, Jax is a more general tool that allows you to continue doing everything by hand from scratch, except the derivative part. Jax will get the gradients for you, as well as jit-compile and parallelize your code to make it blazing fast. You will only need to add one line:
```
dloss = jax.grad(loss)
```
Here are resources to learn Jax:

https://ericmjl.github.io/dl-workshop/01-differential-programming/01-neural-nets-from-scratch.html

https://jax.readthedocs.io/en/latest/

Although I have noted that Jax gives me NaNs when the hand-made numpy NN doesn't. I am actively researching this quandary. Unfortunately the speed boost is worth the pain of finding out where these NaNs come from, because implementing techniques against NaNs have to be done anyways. 

# Step 5: Entire Deep Learning Framework

If you are like me, after creating your own NNs, you will next think "but how does Tensorflow / PyTorch do it?" Well luckily there is a quick online tutorial to make your own customizeable deep learning framework (first link below), and a full on book that teaches you how to make a beefy, capable, no stone left unturned DL framework (second link). The great part is that they both go hard in the math, the code, and intuition. 

https://www.deep-teaching.org/courses/differential-programming

https://www.amazon.com/Deep-Learning-Scratch-Building-Principles/dp/1492041416
