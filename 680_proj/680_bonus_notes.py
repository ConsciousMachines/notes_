









# SIMPLE FONT PREVIEW PROGRAM

DejaVu
Symbola_hint
NotoSerif-Regular
NotoSansMono-Medium
NotoSansDisplay-Regular


# using PyGame instead of tkinter because tkinter renders fonts pixelated

import subprocess 
import pygame

# get a list of the system fonts
# :lang=el                                  <- fonts that support unicode Greek (for math) 
# grep -viE 'italic|bold|condensed|oblique' <- this exclude bold, italic, etc fonts
# cut -d ':' -f 1                           <- this takes out just the file name
# sort -u                                   <- this sorts and removes duplicates
cmd = "fc-list :lang=el | grep -viE 'italic|bold|condensed|oblique' | cut -d ':' -f 1 | sort -u"
# cmd = "fc-list :lang=el | grep -viE 'italic|bold|condensed|oblique' | grep 'DejaVu' | cut -d ':' -f 1 | sort -u"
result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
font_files = [i for i in result.stdout.split('\n') if len(i) > 0]

# the lines to show. I think it's better to actually render something you'd be reading.
lines = []
lines.append('To find the maximum likelihood estimate (MLE) of θ²,')
lines.append('we first need to find the likelihood function, ')
lines.append('then its derivative with respect to θ, ')
lines.append('and finally set the derivative equal to 0 and solve for θ.')
lines.append('Since X₁, ..., Xₙ are independent and identically distributed (i.i.d.) random variables, ')
lines.append('the likelihood function L(θ) is given by the product of probabilities:')
lines.append('L(θ) = P(X₁ = x₁, ..., Xₙ = xₙ | θ) = Πᵢ P(Xᵢ = xᵢ | θ)')
lines.append('Given that Pr(Xᵢ = 1) = θ and Pr(Xᵢ = 0) = 1 - θ, the likelihood function can be written as:')
lines.append('L(θ) = θ^s * (1 - θ)^(n - s)')
lines.append("where s is the number of successes (i.e., the number of 1's) in the sample.")
lines.append('')




# Initialize Pygame
pygame.init()

# Set the screen size
screen_width = 1200
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))

# Define the font size and sample text
font_size = 24

# Load the font files and create a list of font objects
fonts = []
for font_file in font_files:
    font = pygame.font.Font(font_file, font_size)
    fonts.append(font)

# Set the initial font index to 0
font_index = 0

# Run the main loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RIGHT:
                # Switch to the next font
                font_index = (font_index + 1) % len(fonts)
            elif event.key == pygame.K_LEFT:
                # Switch to the previous font
                font_index = (font_index - 1) % len(fonts)
            elif event.key == pygame.K_ESCAPE:
                # Quit the application and clean up resources
                running = False
                pygame.quit()

    # Create a new image with a white background
    _             = img = pygame.Surface((screen_width, screen_height))
    _             = img.fill((91,26,56))

    # Render the sample text using the current font
    font          = fonts[font_index]
    v_offset      = font_size
    lines.append(font_files[font_index]) # add the current font so we know the file name
    for line in lines:
        v_offset  = v_offset + font_size * 1.3
        text      = font.render(line, True, (225,204,140))
        text_rect = text.get_rect(topleft=(font_size, v_offset))
        _         = img.blit(text, text_rect)
    lines.pop() # remove current font file name

    # Update the display
    _             = screen.blit(img, (0, 0))
    _             = pygame.display.flip()

# Clean up
pygame.quit()




















# nth derivative - which i planned to use for taylor series
'''
import torch

def fac(n):
    ret = 1
    for i in range(2, n+1):
        ret *= i
    return ret

# https://stackoverflow.com/questions/50322833/higher-order-gradients-in-pytorch
def nth_derivative(f, wrt, n):
    for i in range(n):
        grads = torch.autograd.grad(f, wrt, create_graph=True)[0]
        f = grads.sum()
    return grads

x = torch.arange(4.).reshape(2, 2)
w = torch.tensor([1., 1.], requires_grad = True).reshape([-1,1])
e = 0.001
v = w - e
y = torch.tensor([0., 1.]).reshape([-1,1])
loss = ((torch.sigmoid(x @ w) - y)**2).sum()

f = loss 
wrt = w
n = 10
for i in range(1, n+1):
    grads = torch.autograd.grad(f, wrt, create_graph=True)[0]
    print(i, grads / fac(i))
    f = grads.sum()

print(nth_derivative(f=loss, wrt=x, n=7))
'''




















'''

ALIGNMENT PROPERTY -> DATA IN SUBSPACE -> LESS DATA FOR GOOD PERFORMANCE

	- ALIGNMENT STRENGTH (% of label norm)

	- NUMBER OF SIGNIFICANT VECTORS

	- STABILITY OF VECTORS
		- When the row space and column space are stable, the singular vectors will be stable as well.
		- perturbation < ||E|| (Davis-Kahan)

	- SINGULAR VALUES DECAY RAPIDLY = MORE STABLE VECTORS
		- more data is concentrated in the few strong vectors vs many noisy ones
		
	- we know we have enough data when the top singular vectors & values emerge and don't change

	- we can extrapolate our findings from the source domain to the target domain by comparing their SVDs
		- compare singular value dropoff
		- compare angles between top singular vectors
		- compare row span of source vs target 
		- how does this predict the degree of transfer of knowledge?

QUESTIONS
- what datasets to use to test these properties? should we switch to NNs?
	- try making a synthetic data: points along a line, add increasing levels of Gaussian noise, 
		compare SVD values & directions.
		this gives us an idea of the stability of singular vectors udner perturbation.
- anything left to do for multiclass case?




- - - 680 project questions for pascal / imani
	
	- - - setup
	x implemented toy example with 2 classes regularization
	x implemented MNIST vs USPS regularization 
	
	- - - multiclass
	x first, check that one-hot vectors still contain label alignment property: yes, they do
	x extend MNIST vs USPS to multiclass: improvement from 20 -> 40 percent (100% improvement)
	o do I need to modify any proofs? I don't think so. nothing else to do?
	
	- - - sample complexity
	- the label alignment property tells us that the data exists in a low dimensional subspace
	- therefore we should require less data to achieve good accuracy
	- 1. study relationship between more data and SVD
		for our case, we are investigating problems that have the alignment property. 
		generate synthetics datasets that exhibit these properties. measure performance of models:
		
		- do singular values decay rapidly / slowly?

		- Alignment strength: how much of label norm is captured in projection. more norm => less data?
		> measure projection norm, good if it's closer to 1 

		- Number of significant singular vectors: majority of projected norm in few vecs => less data?
		> how many vecs contain most of the projection norm?

		- Stability of singular vectors: do singular vecs change significantly when we add new data, or data distribution shifts?
		> see below
		- matrix B has size infinity by d. All rows>n are zero. same SVD as A which is n by d.
			proof:
				A^T A = B^T B by construction
				their covariance matrices are the same, so have the same eigenvalues
				thus the singular values are the same, since they are sqrt of eigenvalues
				the row space is the same -> right singular vectors are the same
				left singular vectors are the same because they have to reconstruct the same columns, with padded zeros
				if the left singular vectors were non-zero after row n, reconstructing columns of B makes it non-zero -> contradiction
		- adding new data points corresponds to a matrix E, which is zeros in first n rows
		- additional data matrix E can be interpreted as a perturbation
		- we can bound the change in eigenvalue, and the change in angle of eigenvector based on ||E||
			https://arxiv.org/pdf/1004.2000.pdf
			https://mathoverflow.net/questions/373616/perturbation-bound-for-svd-denoising-for-a-low-rank-matrix
			https://trungvietvu.github.io/notes/2020/DavisKahan
			
		- when adding more points does not change the row space, the right eigenvectors are stable -> left vecs stable too

	- one thing the paper does not expand on is how the choice of k vs k' is made. 
		- k  is the number of top singular vectors of source
		- k' is the number of top singular vectors of target
		- i propose that we can study the relationship between the SVD of source & target:
			1. how does source label align with source singular vectors?
				1. how much of the label vector is in their span
				2. how many vectors contain most of the label
				3. 
			2. how does the SVD of source compare to SVD of target?
				1. compare singular value dropoff
				2. compare rowspace of data, corresponding to right singular vectors
				3. compare the subspaces spanned by left singular vectors 
					(angles between the vectors)
					(how much of the source span is in the target span)
			* * * what data set should be used for such analysis? Should we switch to NNs?


- - - 680 paper key points
	- unsupervised domain adaptation: 
		transfer knowledge from one domain with labeled data, to another domain with unlabeled data. 
	- domain shift:
		difference in the 2 domains. which prevents the transfer from happening.



- - - SVD by Steve Brunton
	
	- X = [x1 x2.. xm] = U E Vt 
	
		x1 is a face picture turned into column vector
			
			there are m faces, each is a vector of length n
		
		U is the basis for X (and for Rn), so column vector u1 would be eigenface
		
			its size is nxn
		
		E is energy or importance 
		
		Vt is weights
		
			its size is mxm
		
		U,V are unitary meaning UUt = I, UtU = I, and they are rotations

		so V would usually be the weights to reconstruct the face using the left singular vecs.
			but because of the transpose, the first column of Vt becomes the weights of the 
			first singular vec needed in each face. 
			second column is the weights of the second singular vec needed for each face. and etc.
			
			for flow fields, column of Vt is how much of the first vector is in each snapshot, and
			since they are ordered, it shows how it changes over time -> time series
		



- - - 680 project notes
	
	- unsupervised domain adaptation
	
		- a technique that transfers knowledge from one domain with labeled data, to another domain with unlabeled data.
	
		[so we train a model on a labeled data set, and then use the model on another dataset]
	
		[for example, train on cats & dogs, then use on tigers vs boars, which are unlabeled]
	
	- domain shift is the actual difference between the domains
	
		[it prevents a smooth transition from the first labeled dataset to the second unlabeled dataset]
	
		[so the tigers are not perfectly classified using cat features (they activate some dog features too)]
		
	- existing methods try to learn representations that are invariant to the shift
		
		[so learn features of cat fur and then tiger fur activates the same features]
		
		- label shift
		
			- distribution of labels in source/target domains are different, even though relationship of features->labels
				is the same. 
				
			- For example, a model trained on 80/20 cats/dogs is transferred to a dataset 30/70. The relationship between
				pixels to labels is the same. The model will be biased to outputting cats because it saw more cats in 
				the source domain. 
				
		- existing methods may fail when there is label shift (apparently?). authors propose new method, based on LABEL ALIGNMENT
	
	- label alignment
		
		- labels tend to align with the top singular vectors of the data matrix.
		
	- instead of learning invariant representations (old boomer method), we:
	
		- REGULARIZE CLASSIFIER TO ALIGN PREDICTIONS WITH TOP SINGULAR VECTORS 
		
			- "align" is not explained, but the labels should hold this relationship with TSV's 
			
			- these are the principal components, directions of largest variation, and represent the main structure of the data
				
				 
- - - section 2 reading

	- U is a basis for Rn. yU is the projection of y onto these basis vectors. Hence U yU = y.
		the components of yU, yUi, are scalars, that when multiplied by ui, recreate y.
		
	- label alignment 
		is a relationship between the labels and the data-matrix, where variation in the labels is mostly along the top 
		principal components of the data. 
		
		- a dataset has label alignment with rank k if yUi ui = 0 for i>k
			[so yU is orthogonal to ui, making yUi = 0]
			[we cant have ui = 0 since it is a basis vector that spans Rn]
			[so basically yU is in the span of the first k singular vecs]
			
		- y is a vector of +-1's
		
		- k(0.1) means we throw out dimensions after dimension k, such that the thrown out dimensions make up < 0.1 of the total norm
			
			[basically the first k vectors make up 0.9 of the norm]
			
	- ok they in paragraph "Note that the last term": they say the first term is linear regression on a smaller subspace and the 
		second term implicitly enforces label alignment. I dont trust this - especially because the k might be different - so what
		if we actually include y in this second term, without assuming it is zero?
		ok maybe im retarded since it seems the first k principal components get close to y, while the second term seems to make 
		them go to zero. 
		ok in that case, why do we even bother? why not just make the matrix have k dimensions, and not have to minimize d-k of them?
		ok im retarded, the point is that it doesnt use Y so we can use it in domain adaptation.
		
		is it a good idea to completely remove the source domain regularization? why not use both at 50%?
		
		is there any way to get rid of y_hat in the optimality bound theorems?
		
		isnt label alignment just lienar regression?
		
		ok the regularization enforces that prediction vector p is orthogonal to d-k vectors of u_i, and in the span of the first k. 
			but what about the other n-d vectors that are not considered?
			yes that is an issue to consider. but measure how important it is: is most (about 90%) of the norm of Y reocnstructed by U?
			
			
			
- - - appendix A
	[apparently label alignment emerges when a large number of features are correlated with the labels]

	
- - - MAIN TAKEAWAY THUS FAR

	- real world data has the labels highly correlated to k features, which describe 0.9 of the norm of Y
	
	- putting SVD formula into linear regression, and assuming most Y's are 0, we get two terms:
		- one makes the weights close to yU 
		- second makes the weights small for dimensions >K+1

- - - second reading summary

- this work relies on the label slignment property:

	property of the data that the label vector is in the span of the top singular vectors of the data matrix.
	
- ok please remember that PCA is a rotation. So the top singular vectors are simply directions where the 
	data subspace has most variation. ITS NOT SOME WEIRD LINEAR COMBINATION! if the label vector is in the 
	span of the TSV's, then it is a linear regression problem in this subspace. This means the other 
	dimensions are mostly irrelevant. Now, what does this say about the original columns of the data?
	Well, nothing really, we just observe them like that. But the "true" data is the PCA subspace. 
	(by subspace I mean the truncated, most important k components, and thus directions)
	
	ok so the data can happen to live in a low dimensional subspace. The labels are arbitrary. 
	label alignment property says the labels are in the span of the TSV's. They just so happen to be for
	a bunch of real world datasets. 
	
- Y is a linear combination of the top k simngular vectors.
	Y is orthogonal to the rest of them. 



- - - - - Section 2 summary 

Many real world data sets have a property where the label vector Y is mostly in the span of the top few singular vectors of the data matrix X. Y is a vector of length n, and X is a matrix of dimension n by d. So if data matrix X has the singular value decomposition U S V^T, then Y will be mostly in the span of the first k columns of U, where k < d. This can be written as u_i^T Y = 0 for i > k, where u_i is the i^th column of U, and u_i^T is the transpose of u_i. This says Y is orthogonal to the singular vectors u_i for i > k, since their dot product is zero. 

Now consider the linear regression objective: minimize ||X w - Y||^2 with respect to w. 

Let us replace X with its singular value decomposition, we get: ||U S V^T w - Y||^2. 

Since U is a unitary matrix, so is U^T, we can multiply our objective by U^T without changing its value: ||U^T U S V^T w - U^T Y||^2.

Since U^T U = I, the identity matrix, we get: ||S V^T w - U^T Y||^2.

this becomes sum_{i=1}^d (s_i v_i^T w - u_i^T Y)^2 where u_i is the i^th column vector of U, and v_i^T is the i^th row vector of V^T.

Remember that u_i^T Y = 0 for i > k, so we can decompose this into: sum_{i=1}^k (s_i v_i^T w - u_i^T Y)^2 + sum_{i=k+1}^d (s_i v_i^T w)^2

The first term of this equation is a linear regression in a smaller subspace. The second term can be interpreted as the regularization that enforces label alignment. 

- - - gpt-4 break
If our prediction is written p = X w, can you explain why our prediction p will be mostly in the span of the first k singular vectors u_i? 
<GPT-4 was able to reconstruct the proof below, when I asked it to explain stuff I didn't understand, and to include math formulas>
- - - 

Here is the explanation (as per the paper):

our prediction is: p = X w

we replace X with its singular value decomposition: p = U S V^T w

we multiply both sides by U^T: U^T p = U^T U S V^T w 

once again we know U^T U = I, so we get: U^T p = S V^T w 

consider each component: u_i^T p = s_i v_i^T w 

since we minimize s_i v_i^T w for i > k, thus u_i^T p will be minimized, making p orthogonal to the singular vector u_i for i > k. That means p will be mostly in the span of the first k singular vectors u_i, just like the real label vector Y. This is called label alignment regulariation. 

- - - - - Section 3 summary

Let us call the term sum_{i=k+1}^d (s_i v_i^T w)^2, the "label alignment regularization term" because as you said, the prediction will have smaller components along the less informative singular vectors and will be mostly in the span of the first k singular vectors u_i.

Please write a Python function I can use to check whether my label vector Y is mostly in the span of the top few singular vectors of the data matrix X. Use the numpy package, and Y is a vector of length n, while X is a matrix of size n by d. 

'''






#############################################################################################
#############################################################################################
#############################################################################################
#############################################################################################

# - - - 680 project road map 
#     x - implement toy example w regularization
#     x - implement MNIST vs USPS w regularization
#     
#     - - - extending to multi-class
#     x - check that one-hot vectors have label alignment property
#     x - extend MNIST vs USPS to multiclas
#     o - extend proof in appendix A to multiclass 
#
#     - - - sample complexity
#     o - go over the sample complexity discussion & see what i can do 


# As for your question about whether sample complexity can apply to your problem where the label vector aligns with the top singular vectors of the data matrix, the answer is yes. In this case, the sample complexity could be lower than in a more general scenario, as the alignment of the label vector with the top singular vectors indicates that the most important information for predicting the labels is concentrated in a low-dimensional subspace. This simplifies the learning task and could allow the model to achieve good performance with fewer samples.

# toy example 1: 2D data rotated in n-dim space. 

# what makes a problem more complex? How does a problem become more complex, ie require more data?

# probably a larger dimension subspace = more samples (generate toy example to show this)
# also, probably the linearity? or maybe put a NN in previous layer to make the features linear? 

# i can try variant example datasets to study how many samples are needed to get a certain level of accuracy:
# compare: increase/decrease dimensionality of subspace 
# compare: high vs low dimensionality (keeping the subspace dimension constant)
# compare: linear model vs nn basis function (also whats up with hidden representations? see 1st paper)
# compare: more/less noise












'''

- - - things that affect sample complexity
    - model complexity (nn vs linear reg) (how complex is the decision boundary we need to learn?)
    - noise / outliers 
    - difficulty of task: faces vs digits, faces have more complex structure
    - dimensionality of data
    - underlying structure of data can be in a low dimensional manifold

- does sample complexity apply to problems where the label vector aligns with the top singular vectors of the data matrix?
- Yes. In this case, the sample complexity can be lower than a general case, because alignment of the label vector with the top singular vectors indicated that the most important information for predicting labels is mostly in a low-dimensional subspace. This can simplify the learning objective and allow the model to learn the underlying structure with fewer samples. We must set some other variables constant such as the model architecture, the optimization algorithm (Adam), and the hyperparameters. There is no way to measure complexity and give a number, but we can compare two problems as being more or less complex. 

    - generate example data sets to measure relative performance in different situations
TODO: example datasets at 
"Sure, we can generate example datasets based on the factors I mentioned "


- for our case, we are investigating problems that have the alignment property. 
    generate synthetics datasets that exhibit these properties. measure performance of models:

    - Alignment strength: how much of label norm is captured in projection. more norm => less data?
    > measure projection norm, good if it's closer to 1 

    - Number of significant singular vectors: majority of projected norm in few vecs => less data?
    > how many vecs contain most of the projection norm?

    - Stability of singular vectors: do singular vecs change significantly when we add new data, or data distribution shifts?
    > see below

    
- - - stability of singular vectors 
    - When you add new data points or the data distribution shifts, the singular vectors can change. 
    - stability of singular vectors is related to the geometry of the data, as well as the nature of the underlying distribution. 
        One way to think about it is that when the data is well-distributed and the underlying structure is robust, 
        the singular vectors will be more stable.

    - The row and column spaces of the data matrix are indeed related to the stability of singular vectors. 
        When the row space and column space are stable, the singular vectors will be more stable as well.


    - if the singular values decay slowly, it indicates that the singular vectors corresponding to 
        the smaller singular values still capture some useful information about the data, and perturbations in the data matrix can 
        significantly affect the singular vectors. 
        
    - if the singular values decay rapidly, the singular vectors corresponding to the smaller singular values capture less information 
        about the data, and perturbations in the data matrix are less likely to affect the singular vectors significantly.

    - to make sure the singular vectors remain stable, it's important to have a good representation of the underlying distribution of the data. 
        If the data is sampled from the same distribution, even when new data points are added, the row and column spaces will remain 
        relatively stable, leading to more stable singular vectors. 
        Moreover, pre-processing techniques, such as normalization and feature scaling, can improve the stability of singular vectors.

    - One way to analyze the stability of singular vectors is to look at the perturbation theory for singular value decomposition (SVD). 
        it examines how small changes in the data matrix affect the singular vectors and singular values. 

        This can be done by examining the bounds on the changes in singular values and vectors in response to matrix perturbations. 
        A key reference for studying perturbation theory for SVD is the book "Matrix Computations" by Gene H. Golub and Charles F. Van Loan.


    - if the data is spread out along some line in space, the singular vector in that direction will likely be more stable. 
        This is because the dominant singular vector captures the direction of maximal variance, 
        and if the data is well-distributed along a line, the variance in that direction will be consistently high.

    - In the case of data that is spherical along a subspace, the singular vectors may indeed be less stable. 
        If the data is uniformly distributed within the sphere, any rotation of the singular vectors will still represent the same subspace. 
        In this case, the singular vectors are not uniquely determined and can be affected by small perturbations in the data.

    - see Steve Brunton for singular value thresholding, related to ideas from information theory, such as entropy and mutual information.

    - most real-world datasets are not perfectly linear. In these cases, you can still use SVD to find the best linear approximation 
        of the data by projecting it onto a lower-dimensional subspace. This can be helpful for tasks like dimensionality reduction, 
        data compression, and noise reduction.



When discussing bounds on the changes in singular values and vectors due to matrix perturbations, we typically refer to both left and right singular vectors. Perturbation analysis helps us understand how much the singular values and vectors can change when the input matrix is altered slightly. In many practical situations, data is subject to small perturbations, such as noise or errors in measurements. Understanding the behavior of the SVD under these perturbations can be important for the stability of algorithms that rely on SVD.

The singular value decomposition of a matrix A is given by:

A = UΣV^T

where U and V are unitary matrices containing the left and right singular vectors, respectively, and Σ is a diagonal matrix containing the singular values.

Let's consider a perturbed matrix A + E, where E is a small perturbation matrix. The perturbed matrix has the SVD:

A + E = U'Σ'V'^T

Here, U' and V' are the perturbed left and right singular vectors, and Σ' is the perturbed singular values.

One useful result from perturbation analysis is the bounds on the changes in singular values. For any i, the difference between the original and perturbed singular values is bounded by the matrix norm of the perturbation:

|σ_i - σ'_i| ≤ ||E||

Similarly, the bounds on the changes in left and right singular vectors can be expressed using the concept of canonical angles. Let θ_ij be the angle between the i-th original left (or right) singular vector and the j-th perturbed left (or right) singular vector. Then, the sine of the canonical angles can be bounded by the matrix norm of the perturbation:

sin(θ_ij) ≤ ||E|| / σ_i

These bounds give us an understanding of how much the singular values and vectors can change due to small perturbations in the matrix. In general, the left and right singular vectors are more sensitive to perturbations when the corresponding singular values are small. This is because small singular values indicate that the matrix is nearly singular or rank-deficient, making the associated singular vectors less stable under perturbations.


- - - how can adding data to the data matrix be related to the norm of the pterturbation?


When you add additional data points to the data matrix, you are essentially perturbing the matrix by extending its row space. Let's consider the following scenario:

You have a data matrix A of size n x d, where n is the number of data points and d is the dimensionality of each point.
You add m new data points, resulting in an extended data matrix A' of size (n+m) x d.
To analyze the perturbation due to adding new data points, you can create a perturbation matrix E of size (n+m) x d as follows:

The first n rows of E are all zeros, matching the original rows of A.
The last m rows of E contain the new data points.
In this case, the perturbed matrix A + E is equal to the extended data matrix A'. The matrix norm of E, ||E||, represents the "magnitude" of the perturbation caused by adding the new data points.

However, this approach does not capture the changes in the singular vectors directly because the perturbation matrix E has a different row space than A. Instead, you can compute the SVD of the extended data matrix A' and compare its singular values and vectors with those of the original matrix A to understand the effects of adding new data points.

As you add more data points, the singular values and vectors of A' may change. In general, if the new data points are aligned with the existing structure in the data, the dominant singular values and vectors will remain stable, and the corresponding subspace will be more robust to the addition of new data points. On the other hand, if the new data points introduce new structures or directions in the data, the singular values and vectors may change significantly. The extent to which the SVD is affected by adding new data points depends on the geometry of the data and the relationships between the old and new data points.


- - - Can't we use the right singular vectors to compare A and A', since the right singular vectors of both matrices will have the same dimensions d by d?


Yes, you can compare the right singular vectors of A and A' as they both have the same dimensions (d x d). This comparison will give you an idea of how the addition of new data points has affected the relationships between the features (dimensions) in the data.

To do this, you can compute the SVD of both A and A', and then compare their right singular vectors V and V':

Compute the SVD of A, resulting in U_A, S_A, and V_A.
Compute the SVD of A', resulting in U_A', S_A', and V_A'.
Compare the right singular vectors V_A and V_A' by calculating the difference, e.g., ||V_A - V_A'||, or by computing the angle between the corresponding right singular vectors.
Keep in mind that comparing right singular vectors is focused on the relationships between features (dimensions) in the data, rather than the relationships between data points. However, if the addition of new data points significantly changes the structure in the data, you will likely observe differences in the right singular vectors as well.

It's important to note that the stability of the right singular vectors is related to the stability of the left singular vectors, as the left and right singular vectors are connected through the singular values in the SVD. If the dominant left singular vectors remain stable, it is likely that the corresponding right singular vectors will also remain stable.


- - - What exactly is the relation between the stability of the left singular vectors and stability of the right singular vectors? Please include any mathematical formulas, if they help.


The stability of the left singular vectors and right singular vectors is connected through the singular values in the SVD. Recall that the SVD of a matrix A is given by A = UΣV^T, where U and V are orthogonal matrices containing the left and right singular vectors, respectively, and Σ is a diagonal matrix containing the singular values.

When a matrix A is perturbed by a small change ΔA, the perturbed matrix A' = A + ΔA also has an SVD, given by A' = U'Σ'V'^T. To understand the relation between the stability of the left and right singular vectors, we can examine how the singular values and singular vectors change as a result of this perturbation.

The Davis-Kahan theorem provides a bound on the difference between the singular subspaces of A and A' that is related to the perturbation size and the separation between the singular values. For a pair of singular values σ_i and σ_{i+1}, let μ = min{|σ_i - σ_j| : j ≠ i}. Then, the theorem states that:

||sin(Θ(U_i, U'_i))|| ≤ ||ΔA|| / μ
||sin(Θ(V_i, V'_i))|| ≤ ||ΔA|| / μ

Here, Θ(U_i, U'_i) and Θ(V_i, V'_i) denote the angles between the i-th left and right singular vector pairs of A and A', respectively. The theorem shows that the difference between the singular vector pairs is bounded by the perturbation size ||ΔA|| and inversely proportional to the separation μ between the singular values.

From this theorem, we can conclude that if the singular values are well-separated (i.e., μ is large), the corresponding left and right singular vectors will be relatively stable under small perturbations. Conversely, if the singular values are close together, the left and right singular vectors may change significantly even for small perturbations.

In summary, the stability of the left singular vectors and the right singular vectors is related through the singular values. If the dominant singular values are well-separated, both the left and right singular vectors corresponding to those singular values will be relatively stable under small perturbations.


- - - How does data that is not perfectly linear tie into all this? will it have less stable singular vectors, since new data points are likely to perturb the structure of the data? Also, what is a good synthetic data set we can construct to study the effect of perturbations from additional data? what kind of analysis should we perform on the singular vectors?


Data that is not perfectly linear will generally have less stable singular vectors than perfectly linear data because the spread of the data is not concentrated along a single direction. New data points can indeed perturb the structure of the data, affecting the singular values and vectors. However, if the data still exhibits a strong underlying structure, the most dominant singular vectors might remain relatively stable.

To construct a synthetic dataset to study the effect of perturbations from additional data, you can create a dataset that is initially linear and then add varying levels of noise. Here's an example of how to create such a dataset:

Generate a linear dataset by sampling points along a line.
Add Gaussian noise to the data points, gradually increasing the variance of the noise to create different levels of perturbation.
Here's a code example using numpy to generate such a dataset:

import numpy as np
import matplotlib.pyplot as plt

def angle_between_vectors(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    theta = np.arccos(np.clip(cos_theta, -1, 1))
    return np.degrees(theta)

n_points = 100
noise_levels = [0, 0.1, 0.5, 1]

x = np.linspace(0, 10, n_points)
line = 2 * x + 3

fig, axes = plt.subplots(1, len(noise_levels), figsize=(20, 5))

for i, noise_level in enumerate(noise_levels):
    noise = np.random.normal(0, noise_level, n_points)
    perturbed_data = line + noise
    
    axes[i].scatter(x, perturbed_data)
    axes[i].set_title(f'Noise Level: {noise_level}')
    axes[i].set_xlabel('x')
    axes[i].set_ylabel('y')

plt.show()


To analyze the effect of perturbations on the singular vectors, you can perform the following steps:

Compute the SVD for each dataset (with different noise levels).
Observe the singular values and how they change with increasing noise levels.
Observe the angles between the dominant left and right singular vectors for each dataset and the original linear dataset.
You can calculate the angle between two singular vectors using the dot product and arccosine function:

By analyzing how the angles between the dominant singular vectors change with increasing noise levels, you can gain insights into the stability of the singular vectors under different levels of perturbation. If the angles remain small despite the noise, the singular vectors are relatively stable. Conversely, if the angles increase significantly, the singular vectors are less stable.


'''


# PRIORITIES:
# # 2. read the other paper 

# TODO: how does having a source and target density tie into all of this?

'''
if the data has columns that are highly correlated to the label vec, then the label alignment property emerges. 

lemma 3
if the k right singular vectors satisfy ||X v_i|| < e, then X has at most d-k singular values > sqrt(k)*e

prop 4
if X has k columns which dotted with Y have dot product (corr) > 0.8, 
'''












####################### P E R T U R B A T I O N  ############################################
#############################################################################################
#############################################################################################
#############################################################################################

xtr, ytr, xte, yte                = get_data('MNIST', digits)
_X_source                         = add_bias_column(xtr.reshape([-1, 784]))
_X_target                         = add_bias_column(xte.reshape([-1, 784]))
_Y_source                         = ytr.reshape([ytr.shape[0], 1])
_Y_source                         = _Y_source.float()

# check dims, otherwise SVD will not have 784 dims
assert _X_source.shape[0] > _X_source.shape[1], '_X_source not enough data points'
assert _X_target.shape[0] > _X_target.shape[1], '_X_source not enough data points'

# convert labels to 1/-1 for the label align property
labels = torch.zeros([_Y_source.shape[0], len(digits)]) - 1
for i in range(len(digits)):
    labels[(_Y_source == digits[i]).flatten(),i] = 1



A = torch.cat([_X_source, torch.zeros([10000,785])], dim = 0) # extend with 10k zeros (where next points will go)
U, S, Vt, U_, S_, Vt_ = get_svd(_X_source, A)

S[:10].flatten()
S_[:10].flatten()


xte = add_bias_column(xte.reshape([10000, 784]))
Ap = torch.cat([_X_source, xte], dim = 0) # added 10k data points from test set 


















    '''
    # results of basic logistic regression
    >>> scores
    tensor([[0.0000, 1.0000, 0.9784],
            [0.0000, 2.0000, 0.4944],
            [0.0000, 3.0000, 0.3589],
            [0.0000, 4.0000, 0.9293],
            [0.0000, 5.0000, 0.6580],
            [0.0000, 6.0000, 0.7507],
            [0.0000, 7.0000, 0.6610],
            [0.0000, 8.0000, 0.7802],
            [0.0000, 9.0000, 0.8142],
            [1.0000, 0.0000, 0.9780],
            [1.0000, 2.0000, 0.9240],
            [1.0000, 3.0000, 0.9345],
            [1.0000, 4.0000, 0.9670],
            [1.0000, 5.0000, 0.8353],
            [1.0000, 6.0000, 0.7009],
            [1.0000, 7.0000, 0.8937],
            [1.0000, 8.0000, 0.8730],
            [1.0000, 9.0000, 0.9167],
            [2.0000, 0.0000, 0.4879],
            [2.0000, 1.0000, 0.9231],
            [2.0000, 3.0000, 0.7262],
            [2.0000, 4.0000, 0.8731],
            [2.0000, 5.0000, 0.7872],
            [2.0000, 6.0000, 0.5695],
            [2.0000, 7.0000, 0.7374],
            [2.0000, 8.0000, 0.8668],
            [2.0000, 9.0000, 0.6651],
            [3.0000, 0.0000, 0.3589],
            [3.0000, 1.0000, 0.9345],
            [3.0000, 2.0000, 0.7268],
            [3.0000, 4.0000, 0.8484],
            [3.0000, 5.0000, 0.5760],
            [3.0000, 6.0000, 0.5332],
            [3.0000, 7.0000, 0.6411],
            [3.0000, 8.0000, 0.5836],
            [3.0000, 9.0000, 0.5617],
            [4.0000, 0.0000, 0.9293],
            [4.0000, 1.0000, 0.9689],
            [4.0000, 2.0000, 0.8731],
            [4.0000, 3.0000, 0.8490],
            [4.0000, 5.0000, 0.8272],
            [4.0000, 6.0000, 0.5991],
            [4.0000, 7.0000, 0.5474],
            [4.0000, 8.0000, 0.8596],
            [4.0000, 9.0000, 0.6115],
            [5.0000, 0.0000, 0.6571],
            [5.0000, 1.0000, 0.8383],
            [5.0000, 2.0000, 0.7878],
            [5.0000, 3.0000, 0.5766],
            [5.0000, 4.0000, 0.8297],
            [5.0000, 6.0000, 0.4213],
            [5.0000, 7.0000, 0.6412],
            [5.0000, 8.0000, 0.6229],
            [5.0000, 9.0000, 0.6096],
            [6.0000, 0.0000, 0.7541],
            [6.0000, 1.0000, 0.7047],
            [6.0000, 2.0000, 0.5701],
            [6.0000, 3.0000, 0.5356],
            [6.0000, 4.0000, 0.6074],
            [6.0000, 5.0000, 0.4206],
            [6.0000, 7.0000, 0.3592],
            [6.0000, 8.0000, 0.7542],
            [6.0000, 9.0000, 0.5625],
            [7.0000, 0.0000, 0.6571],
            [7.0000, 1.0000, 0.9000],
            [7.0000, 2.0000, 0.7443],
            [7.0000, 3.0000, 0.6473],
            [7.0000, 4.0000, 0.5517],
            [7.0000, 5.0000, 0.6426],
            [7.0000, 6.0000, 0.3555],
            [7.0000, 8.0000, 0.6880],
            [7.0000, 9.0000, 0.5728],
            [8.0000, 0.0000, 0.7762],
            [8.0000, 1.0000, 0.8751],
            [8.0000, 2.0000, 0.8748],
            [8.0000, 3.0000, 0.5849],
            [8.0000, 4.0000, 0.8628],
            [8.0000, 5.0000, 0.6194],
            [8.0000, 6.0000, 0.7497],
            [8.0000, 7.0000, 0.6933],
            [8.0000, 9.0000, 0.7508],
            [9.0000, 0.0000, 0.8100],
            [9.0000, 1.0000, 0.9187],
            [9.0000, 2.0000, 0.6731],
            [9.0000, 3.0000, 0.5666],
            [9.0000, 4.0000, 0.6115],
            [9.0000, 5.0000, 0.6142],
            [9.0000, 6.0000, 0.5577],
            [9.0000, 7.0000, 0.5759],
            [9.0000, 8.0000, 0.7567]])

    >>> torch.mean(scores[:,2])
    tensor(0.7089)
    '''

    '''
    results of using the regularization

    >>> scores
    tensor([[0.0000, 1.0000, 0.3519],
            [0.0000, 2.0000, 0.4601],
            [0.0000, 3.0000, 0.8772],
            [0.0000, 4.0000, 0.6802],
            [0.0000, 5.0000, 0.7876],
            [0.0000, 6.0000, 0.6850],
            [0.0000, 7.0000, 0.9249],
            [0.0000, 8.0000, 0.9341],
            [0.0000, 9.0000, 0.9511],
            [1.0000, 0.0000, 0.3512],
            [1.0000, 2.0000, 0.9145],
            [1.0000, 3.0000, 0.9212],
            [1.0000, 4.0000, 0.9349],
            [1.0000, 5.0000, 0.9189],
            [1.0000, 6.0000, 0.7594],
            [1.0000, 7.0000, 0.9457],
            [1.0000, 8.0000, 0.8624],
            [1.0000, 9.0000, 0.9139],
            [2.0000, 0.0000, 0.4517],
            [2.0000, 1.0000, 0.9240],
            [2.0000, 3.0000, 0.3691],
            [2.0000, 4.0000, 0.8097],
            [2.0000, 5.0000, 0.4365],
            [2.0000, 6.0000, 0.8332],
            [2.0000, 7.0000, 0.6944],
            [2.0000, 8.0000, 0.7465],
            [2.0000, 9.0000, 0.7029],
            [3.0000, 0.0000, 0.8763],
            [3.0000, 1.0000, 0.9269],
            [3.0000, 2.0000, 0.3697],
            [3.0000, 4.0000, 0.8144],
            [3.0000, 5.0000, 0.6143],
            [3.0000, 6.0000, 0.6375],
            [3.0000, 7.0000, 0.7933],
            [3.0000, 8.0000, 0.7937],
            [3.0000, 9.0000, 0.7295],
            [4.0000, 0.0000, 0.6707],
            [4.0000, 1.0000, 0.9392],
            [4.0000, 2.0000, 0.8097],
            [4.0000, 3.0000, 0.8103],
            [4.0000, 5.0000, 0.7985],
            [4.0000, 6.0000, 0.5824],
            [4.0000, 7.0000, 0.5547],
            [4.0000, 8.0000, 0.7853],
            [4.0000, 9.0000, 0.6426],
            [5.0000, 0.0000, 0.7880],
            [5.0000, 1.0000, 0.9144],
            [5.0000, 2.0000, 0.4371],
            [5.0000, 3.0000, 0.6123],
            [5.0000, 4.0000, 0.7985],
            [5.0000, 6.0000, 0.6097],
            [5.0000, 7.0000, 0.5776],
            [5.0000, 8.0000, 0.5836],
            [5.0000, 9.0000, 0.7476],
            [6.0000, 0.0000, 0.6850],
            [6.0000, 1.0000, 0.7451],
            [6.0000, 2.0000, 0.8332],
            [6.0000, 3.0000, 0.6393],
            [6.0000, 4.0000, 0.5836],
            [6.0000, 5.0000, 0.6135],
            [6.0000, 7.0000, 0.6230],
            [6.0000, 8.0000, 0.6907],
            [6.0000, 9.0000, 0.7789],
            [7.0000, 0.0000, 0.9254],
            [7.0000, 1.0000, 0.9423],
            [7.0000, 2.0000, 0.6961],
            [7.0000, 3.0000, 0.7921],
            [7.0000, 4.0000, 0.5547],
            [7.0000, 5.0000, 0.5769],
            [7.0000, 6.0000, 0.6199],
            [7.0000, 8.0000, 0.7600],
            [7.0000, 9.0000, 0.6020],
            [8.0000, 0.0000, 0.9275],
            [8.0000, 1.0000, 0.8609],
            [8.0000, 2.0000, 0.7465],
            [8.0000, 3.0000, 0.7918],
            [8.0000, 4.0000, 0.7853],
            [8.0000, 5.0000, 0.5836],
            [8.0000, 6.0000, 0.6900],
            [8.0000, 7.0000, 0.7613],
            [8.0000, 9.0000, 0.7417],
            [9.0000, 0.0000, 0.9499],
            [9.0000, 1.0000, 0.9081],
            [9.0000, 2.0000, 0.7029],
            [9.0000, 3.0000, 0.7410],
            [9.0000, 4.0000, 0.6408],
            [9.0000, 5.0000, 0.7450],
            [9.0000, 6.0000, 0.7789],
            [9.0000, 7.0000, 0.6020],
            [9.0000, 8.0000, 0.7417]])

    >>> torch.mean(scores[:,2])
    tensor(0.7302)

    fairly similar result to what the paper claimed, about 2-3% improvement.
    '''




# EXPERIMENTS
# general
# no reg: accuracies: .92 / .41    k1=0, k2=96, lam=1000
# k2 = 96, lam 1..10^7 all < .40
# accuracies: .35 / .53    k1=0, k2=13, lam=100000

#accuracies: .60 / .56    k1=15, k2=13, lam=1000000
#accuracies: .54 / .57    k1=0, k2=13, lam=1000000
#accuracies: .82 / .52    k1=0, k2=96, lam=10000000



soy = S_t.cpu().numpy()
plt.plot(np.log(soy))
plt.show()


soy = S_s.cpu().numpy()
plt.plot(np.log(soy))
plt.show()


# https://www.jmlr.org/format/format.html
# https://arxiv.org/abs/2211.14960
# https://openreview.net/forum?id=fLIWMnZ9ij

'''
- ask pascal 
	- i am looking at the open review website for the paper I cited, and reading reviews for inspiration. How long does it take for a person
		to write a review like this?
	- do you undertand the proof in appendi A?
'''










