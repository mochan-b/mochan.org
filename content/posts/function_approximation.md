+++
title = "Linear Function Approximation"
date = 2026-04-24T19:59:37-04:00
draft = false
+++

In reinforcement learning, function approximation is a technique to estimate value functions or policies when the state or action space is too large or continuous to be represented in a simple table. So, instead of storing a discrete value for every possible state, we approximate the function using a set of learned parameters combined with feature extractors. We use features like polynomial bases, Fourier bases, radial basis functions (RBF), or tile coding. By evaluating these features with a linear value function and a linear softmax policy, the agent can efficiently generalize its experiences from visited states to similar unvisited ones.

## Linear Function Approximation

At the core of linear function approximation is the idea of representing a complex, continuous function (such as a value function or a policy) as a linear combination of features. Instead of maintaining a distinct value for every possible state, we define a parameterized function. For a value function $\hat{v}(s, \mathbf{w})$ where $s$ is the state of the environment and $\mathbf{w}$ is the weight vector of the function, this is expressed as the inner product of a weight vector $\mathbf{w}$ and a feature vector $\boldsymbol{\phi}(s)$:

$$ \hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s) = \sum_{i=1}^n w_i \phi_i(s) $$

The vector $\boldsymbol{\phi}(s) \in \mathbb{R}^n$ is produced by a **feature extractor** that maps the raw state $s$ into a representation suitable for learning. The learning algorithm's job is simply to update the weights $\mathbf{w}$ using gradient descent to minimize the error between the predicted value and the target return.

Because the function is linear with respect to the weights $\mathbf{w}$, the gradient is exactly the feature vector itself: 
$$
\nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w}) = \boldsymbol{\phi}(s).$$ This makes optimization extremely stable and computationally efficient. 

The true power of this method relies heavily on how we define the feature extractor $\boldsymbol{\phi}(s)$. Different bases provide different ways to represent the continuous state space:

- **Tile Coding:** A sparse, binary representation using overlapping grids. It provides a highly efficient way to achieve local generalization with minimal computational overhead.
- **Polynomial Bases:** Map the state components to various polynomial degrees (e.g., $s_1, s_1^2, s_1 s_2$). They are simple but can suffer from unbounded growth and global generalization, meaning an update in one local region might drastically change predictions in distant parts of the state space.
- **Fourier Bases:** Use trigonometric functions (sines and cosines) at various frequencies. They are excellent for continuous state spaces and often outperform polynomials because they naturally bound the feature values to $[-1, 1]$.
- **Radial Basis Functions (RBFs):** Place a grid of Gaussian bells over the state space. A state activates multiple RBFs based on its distance to their centers. They offer excellent smooth local generalization but require computing exponentials, which can be computationally heavy compared to simpler methods.

## Tile Coding

Tile coding is a form of coarse coding that provides a sparse, computationally efficient representation of continuous multidimensional state spaces. The method relies on defining multiple overlapping partitions, or "tilings," of the state space.

![Illustration of 2D Tile Coding with 2 overlapping tilings](/images/tile_coding.svg)

Formally, consider a $d$-dimensional continuous state space $\mathcal{S} \subset \mathbb{R}^d$. We construct $N$ tilings, where each tiling is a regular grid that partitions $\mathcal{S}$ into hyperrectangular bins called "tiles." To ensure generalization across states, each of the $N$ tilings is offset from the others by a uniform spatial displacement vector. In a typical implementation, if each dimension is partitioned into $k$ tiles, the offset along each dimension for the $i$-th tiling is given by $\Delta_i = \frac{i}{N \cdot k}$ so that the tilings are evenly spaced and cover the state space with different alignments. This overlapping structure allows for local generalization: states that are close together will share active tiles across multiple tilings, while states that are far apart will have fewer shared tiles, enabling the representation to capture both local and global structure in the state space.

When a continuous state $s \in \mathcal{S}$ is evaluated, it falls into exactly one tile per tiling. The feature extractor maps $s$ to a binary feature vector $\boldsymbol{\phi}(s) \in \{0, 1\}^D$, where $D = N \cdot k^d$ is the total number of tiles across all tilings. To be precise, $\boldsymbol{\phi}(s)$ is the concatenation of $N$ smaller one-hot vectors, one for each tiling. We can index this full vector using two variables: let $\phi_{i, j}(s)$ denote the component for the $j$-th tile within the $i$-th tiling. Specifically, $\phi_{i, j}(s) = 1$ if the state $s$ lies within the $j$-th tile of tiling $i$, and $\phi_{i, j}(s) = 0$ otherwise. Because exactly one tile is active per tiling, the resulting concatenated feature vector $\boldsymbol{\phi}(s)$ is highly sparse, with exactly $N$ non-zero elements ($\|\boldsymbol{\phi}(s)\|_0 = N$, where the 0-"norm" is defined as the number of non-zero elements).

![Illustration of the concatenated feature vector and gradient](/images/tile_coding_vector.svg)

This structural sparsity is highly advantageous for linear function approximation. Because $\boldsymbol{\phi}(s)$ is a binary vector with exactly $N$ ones, the value function computation $\hat{v}(s, \mathbf{w}) = \mathbf{w}^\top \boldsymbol{\phi}(s)$ simplifies dramatically. If we denote $w_{i, j}$ as the weight associated with the $j$-th tile of the $i$-th tiling, instead of a full vector dot product requiring $D$ multiplications, the computation reduces to a simple sum of the $N$ weights corresponding to the active tiles:

$$ \hat{v}(s, \mathbf{w}) = \sum_{i=1}^N \sum_{j \in \text{active tile}(s, i)} w_{i, j} $$

Similarly, the gradient update during learning becomes incredibly efficient. The gradient $\nabla_{\mathbf{w}} \hat{v}(s, \mathbf{w}) = \boldsymbol{\phi}(s)$ is just the sparse feature vector itself, meaning only the $N$ weights corresponding to the active tiles are updated. The overlapping nature of the grids allows for local generalization—states that are spatially proximate will share active tiles across several tilings, meaning an update to one state will partially update the values of nearby states. Conversely, the uniform offsets between tilings grant the representation fine-grained discriminative resolution, mitigating the discretization error inherent in a single grid approach.

## Polynomial Bases

While tile coding provides a sparse, local representation, polynomial bases offer a dense, global approach. This method relies on the standard polynomial expansion to map the state variables into a higher-dimensional feature space.

For a one-dimensional state $s$, a polynomial feature extractor of degree $n$ simply outputs the powers of $s$:
$$ \boldsymbol{\phi}(s) = \begin{bmatrix} 1, & s, & s^2, & \dots, & s^n \end{bmatrix}^\top $$

![Illustration of 1D polynomial basis functions](/images/poly_basis.svg)

For a multi-dimensional state space, the basis includes the combinations (cross-terms) of the state variables such that the sum of their powers does not exceed the chosen degree $n$. For instance, given a 2D state $s = [s_1, s_2]$ and a degree of $n=2$, the feature vector contains:
$$ \boldsymbol{\phi}(s) = \begin{bmatrix} 1, & s_1, & s_2, & s_1 s_2, & s_1^2, & s_2^2 \end{bmatrix}^\top $$

As the degree $n$ and the state dimensionality $d$ increase, the number of features grows rapidly, following $\binom{n+d}{n}$. In practice, this means we must first **normalize** our state variables (typically into the $[-1, 1]$ or $[0, 1]$ range) before applying the polynomial expansion. Without normalization, raising large state values to high powers will cause the feature values to explode, leading to massive gradients and numerical instability during learning.

The key characteristic of polynomial bases is **global generalization**. Because a term like $s^2$ is non-zero almost everywhere, changing the weight associated with $s^2$ to correct the value prediction in one specific state region will intrinsically alter the predictions across the entire state space. This can make learning challenging in complex environments, as updates in one area can "unlearn" accurate predictions in a distant area. Due to this global interference, polynomial bases are often less favored in modern reinforcement learning compared to Fourier bases or local representations like Tile Coding.

## Fourier Bases

Fourier bases provide a robust alternative to polynomials for continuous state spaces. Instead of using increasing powers of the state variables, this method represents the state using trigonometric functions—specifically, a series of cosines at varying frequencies.

Before applying the Fourier basis, the continuous state space is strictly normalized to the $[0, 1]$ range. For a $d$-dimensional normalized state $s \in [0, 1]^d$, the basis functions are defined by a set of coefficient vectors $\mathbf{c} \in \mathbb{Z}^d$. Each component of the feature vector $\boldsymbol{\phi}(s)$ is computed as:

$$ \phi_{\mathbf{c}}(s) = \cos(\pi \mathbf{s}^\top \mathbf{c}) $$

where $\mathbf{s}^\top \mathbf{c}$ is the dot product of the state vector and the coefficient vector. The coefficients determine the frequency of the cosine wave along each state dimension. In practice, we define a maximum order $n$ and generate all possible coefficient vectors where each element $c_i \in \{0, 1, \dots, n\}$.

![Illustration of 1D Fourier basis functions](/images/fourier_basis.svg)

For example, in a 1D state space with order $n=3$, the coefficients are $c \in \{0, 1, 2, 3\}$, resulting in the feature vector:
$$ \boldsymbol{\phi}(s) = \begin{bmatrix} \cos(0), & \cos(\pi s), & \cos(2\pi s), & \cos(3\pi s) \end{bmatrix}^\top $$
Notice that the first feature is always $\cos(0) = 1$, serving as a bias term, just like in the polynomial basis.

Fourier bases offer several key advantages over polynomial bases. First, trigonometric functions are naturally bounded between $[-1, 1]$. No matter how complex the environment or how high the order $n$, the feature values will never explode, making gradient updates substantially more stable. Second, Fourier series are exceptionally well-suited for approximating periodic or smooth continuous functions, often requiring a lower degree $n$ to achieve the same accuracy as a polynomial expansion. While they still exhibit global generalization (changing a weight affects the entire cosine wave), their bounded nature and frequency-based decomposition make them highly effective for linear function approximation in reinforcement learning.

## Radial Basis Functions (RBFs)

If Tile Coding is a rigid, binary approach to local generalization, Radial Basis Functions (RBFs) provide a continuous, smooth alternative. Instead of discrete square tiles, an RBF network places a grid of soft "Gaussian bells" over the continuous state space.

Each feature in an RBF network corresponds to a specific center point $\mathbf{c}_i \in \mathbb{R}^d$. The activation of the $i$-th feature depends entirely on the distance between the current state $s$ and the center $\mathbf{c}_i$. The standard Gaussian RBF is defined as:

$$ \phi_i(s) = \exp\left( -\frac{\|s - \mathbf{c}_i\|^2}{2\sigma^2} \right) $$

where $\|s - \mathbf{c}_i\|^2$ is the squared Euclidean distance, and $\sigma$ (sigma) is a bandwidth parameter that controls the "width" of the bell. A small $\sigma$ creates narrow bells (highly localized), while a large $\sigma$ creates wide, overlapping bells (broader generalization).

![Illustration of 1D Radial Basis Functions](/images/rbf_basis.svg)

In practical reinforcement learning implementations, it is common to **normalize** the RBF activations so that they sum to $1$ across all features for any given state. This ensures that the total activation magnitude remains stable, preventing the value function from outputting drastically different magnitudes in regions where bells overlap heavily versus regions where they are sparse. The normalized feature is given by:

$$ \tilde{\phi}_i(s) = \frac{\phi_i(s)}{\sum_{j=1}^N \phi_j(s)} $$

Unlike Polynomial or Fourier bases, which alter the function globally, RBFs offer **smooth local generalization**. Because the exponential function approaches zero as distance increases, a state will only significantly activate the few RBF centers closest to it. Updating the weights for those active features will smoothly adjust the value function in that specific local neighborhood without disturbing predictions on the opposite side of the state space.

However, RBFs come with two notable downsides. First, computing distances and exponentials for every center at every time step is computationally expensive compared to evaluating simple polynomials or discrete tile indices. Second, like Tile Coding, RBFs suffer from the *curse of dimensionality*. If you want 5 centers per dimension in a 4-dimensional state space, you need $5^4 = 625$ centers. In an 8-dimensional space, you would need nearly 400,000 centers, making a uniform grid of RBFs impractical for high-dimensional problems without advanced techniques to selectively place the centers.

## Basis Function Choices for CartPole

To see how these abstract concepts translate into practice, let's look at the specific parameters we chose to train our Actor-Critic agent on the classic CartPole environment. CartPole features a 4-dimensional continuous state space ($d=4$): cart position, cart velocity, pole angle, and pole angular velocity. Before applying any of these methods, we normalize all four dimensions to the $[0, 1]$ range.

Depending on which feature extractor we select, we have to make explicit design choices regarding their hyperparameters. These choices directly govern the size of our linear weight vectors and the resolution of our agent's learning.

### Tile Coding Parameters

For Tile Coding, we chose **8 tilings** (`n_tilings=8`) and **4 tiles per dimension** (`n_tiles=4`). 

Since our state space is 4D, each individual tiling contains $4^4 = 256$ bins. Overlapping 8 of these grids yields a total feature vector size of $D = 8 \times 256 = 2048$. While 4 bins per dimension might seem like a very coarse resolution for a continuous variable, the 8 uniformly shifted tilings work together to provide fine-grained discriminative power. At any given time, exactly 8 elements out of the 2048 will be active (a `1`), meaning our gradient update touches just 8 weights per step, making it incredibly fast.

### Polynomial Basis Parameters

For the Polynomial Basis, we chose a **degree of 3** (`degree=3`).

This means our feature vector includes all combinations of our 4 state variables where the sum of their exponents does not exceed 3 (e.g., $s_1^2 s_2$, or $s_1 s_2 s_3$). In a 4D space, a degree-3 polynomial generates $\binom{4+3}{3} = 35$ distinct features. While a 35-element weight vector is highly compact and memory-efficient, the global generalization inherent to polynomials means the agent will likely struggle with interference during learning compared to the local methods.

### Fourier Basis Parameters

For the Fourier Basis, we chose an **order of 3** (`order=3`).

This defines our coefficient vectors $\mathbf{c}$. Because each of the 4 dimensions can independently take a coefficient $c_i \in \{0, 1, 2, 3\}$, we generate $4^4 = 256$ possible frequency combinations. This gives us a dense 256-element feature vector of bounded cosine waves. This provides a sufficiently rich frequency domain to smoothly approximate the CartPole value function and policy without the exploding gradients typical of polynomials.

### Radial Basis Function (RBF) Parameters

For the RBF network, we chose **5 centers per dimension** (`n_centers_per_dim=5`) with a **bandwidth of $\sigma=0.3$** (`sigma=0.3`).

Laying down 5 centers evenly across each of the 4 dimensions creates a uniform grid of $5^4 = 625$ Gaussian bells covering the state space. A bandwidth of $\sigma=0.3$ (relative to our normalized $[0, 1]$ space) ensures that the bells overlap generously. Because we normalize the activations, a state resting exactly between two centers will smoothly split its activation between them, providing continuous local generalization with a 625-element weight vector.

