+++
title = 'Importance Sampling in Off-Policy Reinforcement Learning'
author = 'Mochan Shrestha'
date = 2025-02-08T12:13:27-05:00
draft = false
+++

Importance sampling is a technique that solves a fundamental challenge in RL: how do we learn about one policy while collecting data from a different policy? Data collection can expensive and time-consuming, so being able to reuse the data we already have to keep training a new policy is very valuable. If we had to sample new data from a policy every time we wanted to make an update for an algorithm, it might become impractical to train anything useful.

More formally, importance sampling is a method in off-policy reinforcement learning (RL), allowing for the estimation of expectations under one distribution while using samples collected from another. Off-policy learning enables the reuse of past experiences, making learning more sample-efficient and reducing the need for on-policy data collection. 

In reinforcement learning, we want to maximize the expected return which is the objective function 
$$
    J(\theta) = \mathbb{E}_{\tau \sim \pi_{\theta}} [R(\tau)]
$$
where $\tau$ represents trajectories sampled from the policy $\pi_{\theta}$. $R(\tau)$ is the cumulative reward for trajectory $\tau$. If we're using the data from the old policy $\pi_{old}$, our objective becomes $\mathbb{E}_{\tau \sim \pi_{old}} [R(\tau)]$ and there is no $\theta$ in the equation and so no gradient with respect to $\theta$ and learning won't happen.

The technique we use in importance sampling where we are still using trajectories from the old distribution but want to improve the new policy by reweighing the returns. The problem is that the actions taken under the old policy may not align with the new policy and the returns may not accurately reflect the performance of the new policy. Thus, the intuition behind reweighing the returns is to give more importance to those trajectories that are more likely under the new policy and less importance to those that are less likely. This way, we can estimate the expected return of the new policy using data collected from the old policy. 

The mathematical basis for this is a change of measure where we can change the probability measure from $P$ to $Q$ using the Radon-Nikodym derivative $\frac{dP}{dQ}$. This gives us the relationship:
$$
    \mathbb{E}_P[f] = \int f \\ dP = \int f \cdot \frac{dP}{dQ} d{Q} = \mathbb{E}_Q\left[f \frac{dP}{dQ}\right]
$$
For the above equation to hold, $P$ and $Q$ must be absolutely continuous with respect to each other. In other words, if an event has zero probability under $Q$, it must also have zero probability under $P$. What this means is that any trajectory that could be sampled under the target policy must also be possible under the behavior policy. Furthermore, the Radon-Nikodym derivative $\frac{dP}{dQ}$ must be well-defined and finite almost everywhere. This ensures that the weighting does not lead to infinite or undefined values, which could destabilize the estimation process. What this means for RL policy is that the behavior policy must have non-zero probability of taking all actions that the target policy might take in any given state and vice versa. 

In the context of RL, $P$ represents the distribution of trajectories under the target policy $\pi$, while $Q$ represents the distribution under the behavior policy $\pi_b$. The Radon-Nikodym derivative $\frac{dP}{dQ}$ corresponds to the importance sampling ratio, which adjusts for the difference between the two policies.

However, there are challenges with importance sampling. 
- The major issue is __high variance__ in the estimates. The denominator of the ratio can become very small if the behavior policy assigns low probability to actions that the target policy favors. This can lead to large importance weights, which in turn can cause high variance in the estimates of expected returns.
- __Weight degeneracy__ is another challenge where a most samples get very low weights, effectively reducing the sample size and just a few samples dominate the estimate. So, even though we have a lot of data from the behavior policy, only a small subset of it effectively contributes to the learning process and results in not actually estimating the expected return accurately.
- Importance sampling is unbiased in theory. However with high variance, the estimates can be very noisy and unreliable in practice, making it difficult to learn a stable policy. So, we do a __bias-variance tradeoff__ to reduce variance at the cost of introducing some bias by using different techniques like normalization of weights or clipping. However, this could lead to a biased estimate of the expected return.
- Another challenge is the __exploration-exploitation tradeoff__. If the behavior policy does not explore certain actions or states sufficiently, the importance sampling estimates may be biased or inaccurate. So, the new policy could get locked into suboptimal behaviors because it never gets data about better actions or states.
- Importance __sampling over trajectories__ can lead to exponentially growing variance with the length of the trajectory. This is because the importance weights are products of per-step ratios, and small discrepancies at each step can compound over time.
- Importance sampling is an estimation tool whereas reinforcement learning is a control problem. Errors can __compound over time__ as the policy is updated iteratively. As the policy changes, the discrepancy between the behavior and target policies can increase, leading to larger importance weights and higher variance in subsequent estimates.

## Importance Sampling Basics

### General Probability

Let's start with the main objective and notations. We want to compute expectations under a distribution (target) defined by $p(x)$. We have simplified the notation from abstract $P$ to concrete $p(x)$ for clarity.
$$
    \mathbb{E}_{x \sim p(x)}[f(x)] = \int f(x) \; p(x) dx
$$
We can only sample from a different distribution $q(x)$ (behavior distribution) and not $p(x)$. As long as $p(x) > 0$ implies $q(x) > 0$, we can use importance sampling to estimate the expectation:
$$
    \mathbb{E}_{x \sim p(x)}[f(x)] = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_{x \sim q(x)}\left[f(x) \frac{p(x)}{q(x)}\right]
$$
and the importance weight or importance sampling ratio is:
$$
    w(x) = \frac{p(x)}{q(x)}
$$
So, the fundamental idea of importance sampling is to calculate expectation of $f(x)$ under $p(x)$ using change of variables to $q(x)$ and reweighing the samples by the importance weight $w(x)$.

### Applying Importance Sampling to Reinforcement Learning

To do reinforcement learning, we would have collected a dataset of experience (trajectories of states, actions, rewards) using a behavior policy $\pi_b$. However, we want to train a different target policy $\pi$. The distribution of trajectories generated by the behavior policy $\pi_b$ differs from that of the target policy $\pi$.

Let us define a trajectory $\tau$ of length $T$ as the following sequence
$$
\tau = (S_0, A_0, R_1, S_1, A_1, R_2, \ldots, S_{T-1}, A_{T-1}, R_T)
$$
where $S_t$ is the state at time $t$, $A_t$ is the action taken at time $t$, and $R_{t+1}$ is the reward received after taking action $A_t$ in state $S_t$.

The return $R(\tau)$ for trajectory $\tau$ is defined as the cumulative discounted reward:
$$
    R(\tau) = \sum_{t=0}^{T-1} \gamma^t R_{t+1}
$$
where $\gamma \in [0, 1)$ is the discount factor.

In order to train our policy $\pi$, we first consider the expected return (value function)
starting from an initial state $s_0$ and following policy $\pi$:
$$
    v_{\pi}(s_0) = \mathbb{E}_{\tau \sim P(\cdot \mid s_0, \pi)}[R(\tau)].
$$
By definition, this expectation can be written as
$$
    v_{\pi}(s_0) = \sum_{\tau} P(\tau \mid s_0, \pi)\, R(\tau),
$$
where $P(\tau \mid s_0, \pi)$ is the probability of trajectory $\tau$ under policy $\pi$ starting from state $s_0$.

For learning in reinforcement learning, we want an objective function $J({\pi})$ which we want to maximize. So, borrowing from our returns and value function, we can write our objective function as
$$
    J(\pi) = \mathbb{E}_{s_0 \sim \rho_0} \left[v_{\pi}(s_0) \right] = \mathbb{E}_{\tau \sim P(\cdot \mid s_0, \pi)}[R(\tau)]
$$
with the gradient given as
$$
    \nabla J(\pi) = \mathbb{E}_{\tau \sim P(\cdot \mid s_0, \pi)} \left[ \nabla \log P(\tau \mid s_0, \pi) R(\tau) \right]
$$
which comes from the likelihood ratio trick, i.e., $\nabla P(x) = P(x) \nabla \log P(x)$ which comes from the chain rule of derivatives using $\nabla \log P(x) = \frac{\nabla P(x)}{P(x)}$.

---

In our case , we have data collected from a behavior policy $\pi_b$ but we want to evaluate or improve a different target policy $\pi$. The distribution of trajectories generated by the behavior policy $\pi_b$ differs from that of the target policy $\pi$. Directly using samples from $\pi_b$ to estimate expectations under $\pi$ would lead to biased estimates.

Under a policy $\pi$, the probability of a trajectory $\tau$ is given by:
$$    P(\tau | \pi) = P(s_0) \prod_{t=0}^{T-1} \pi(a_t | s_t) P(s_{t+1} | s_t, a_t) $$
where $P(s_0)$ is the initial state distribution and $P(s_{t+1} | s_t, a_t)$ is the environment's transition probability.

Now, let us look at the ratio of probabilities of the same trajectory $\tau$ under the target policy $\pi$ and the behavior policy $\pi_b$:
$$
    \frac{P(\tau | \pi)}{P(\tau | \pi_b)} = \frac{P(s_0) \prod_{t=0}^{T-1} \pi(a_t | s_t) P(s_{t+1} | s_t, a_t)}{P(s_0) \prod_{t=0}^{T-1} \pi_b(a_t | s_t) P(s_{t+1} | s_t, a_t)} = \prod_{t=0}^{T-1} \frac{\pi(a_t | s_t)}{\pi_b(a_t | s_t)}
$$
The transition probabilities and initial state distribution cancel out because they are the same under both policies. Thus, the discrepancy between the two policies is captured entirely by the action probabilities and not the environment dynamics.


Thus, given a behavior policy $\pi_b$ and a target policy $\pi$, the importance sampling ratio is defined as:

$$
    w_t = \frac{\pi(a_t | s_t)}{\pi_b(a_t | s_t)}
$$
where:
- $s_t$ is the state at time $t$,
- $a_t$ is the action taken,
- $\pi(a_t | s_t)$ is the probability of taking action $a_t$ under the target policy,
- $\pi_b(a_t | s_t)$ is the probability of taking the same action under the behavior policy.

and for the entire trajectory, the cumulative importance weight is:
$$
    W(\tau) = \prod_{t=0}^{T-1} w_t = \prod_{t=0}^{T-1} \frac{\pi(a_t | s_t)}{\pi_b(a_t | s_t)}
$$

---

Combining the importance sampling ratio and the expected return, we can estimate the value function under the target policy $\pi$ using samples collected from the behavior policy $\pi_b$:
$$
    v_{\pi}(s_0) = \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ W(\tau) R(\tau) \right]
$$
This equation shows that we can compute the expected return under the target policy by reweighing the returns of trajectories sampled from the behavior policy using the importance sampling ratio.

For the objective function, we can similarly write:
$$
    J(\pi) = \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ W(\tau) R(\tau) \right]
$$
and the gradient becomes:
$$
    \nabla J(\pi) = \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ W(\tau) \nabla \log P(\tau | s_0, \pi) R(\tau) \right]
$$

---

__Theorem 1__ : Given a behavior policy $\pi_b$ and a target policy $\pi$, importance sampling provides an unbiased estimator for the expected return under the target policy using samples from the behavior policy, provided that $\pi_b(a | s) > 0$ whenever $\pi(a | s) > 0$ for all states $s$ and actions $a$.

__Proof__: The proof follows directly from the definition of importance sampling. We want to estimate:
$$
    \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi)}[R(\tau)] = \sum_{\tau} P(\tau | s_0, \pi) R(\tau)
$$
Using importance sampling, we can rewrite this expectation using samples from the behavior policy $\pi_b$:
$$
\begin{aligned}
    \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ W(\tau) R(\tau) \right] &= \sum_{\tau} P(\tau | s_0, \pi_b) W(\tau) R(\tau ) \\
    &= \sum_{\tau} P(\tau | s_0, \pi_b) \frac{P(\tau | s_0, \pi)}{P(\tau | s_0, \pi_b)} R(\tau) \\
    &= \sum_{\tau} P(\tau | s_0, \pi) R(\tau) \\
    &= \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi)}[R(\tau)]
\end{aligned}
$$  
Thus, the importance sampling estimator is unbiased. $\qquad \qquad \qquad \qquad \square$

---

__Theorem 2 (Variance of Importance Sampling)__ : Let $\pi_b$ be the behavior policy and $\pi$ be the target policy. The variance of the importance sampling estimator for the expected return under the target policy is given by:
$$
    \text{Var}\left( W(\tau) R(\tau) \right) = \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} [ \left( W(\tau) R(\tau) \right)^2 ] - \left( \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ W(\tau) R(\tau) \right] \right)^2
$$
with the support condition that $\pi_b(a | s) > 0$ whenever $\pi(a | s) > 0$ for all states $s$ and actions $a$ and where $W(\tau)$ is the importance sampling ratio for trajectory $\tau$ given by:
$$
    W(\tau) = \prod_{t=0}^{T-1} \frac{\pi(a_t | s_t)}{\pi_b(a_t | s_t)}
$$
Moreover, if $\pi \neq \pi_b$, then there exists finite-horizon MDPs with bounded rewards for which the variance grows exponentially with the trajectory length $T$. 

Furthermore, if there exists $\epsilon > 0$ such that for
every reachable state $s$, the per-step importance weight has second moment bounded below:
$$
    \mathbb{E}_{a \sim \pi_b(\cdot|s)}\left[ \left(\frac{\pi(a|s)}{\pi_b(a|s)}\right)^2 \right] \geq 1 + \epsilon
$$
then the variance of the importance sampling estimator grows at least exponentially with the trajectory length $T$.

__Proof__: The variance of a random variable $X$ is defined as $\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2$. Applying this definition to the importance sampling estimator $W(\tau) R(\tau)$, we have:
$$
\begin{aligned}
    \text{Var}\left( W(\tau) R(\tau) \right) &= \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} [ \left( W(\tau) R(\tau) \right)^2 ] - \left( \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ W(\tau) R(\tau) \right] \right)^2
\end{aligned}
$$

---

Now, let us construct a simple MDP to show that the variance can grow exponentially with the trajectory length $T$ when $\pi \neq \pi_b$. Let us consider a single state MDP bandit problem where there is only one state and two actions $a$ and $b$.

It has the following properties:
- Reward is always 1 per step and so, $R(\tau) = T$ for any trajectory $\tau$ of length $T$.
- Behaviour policy $\pi_b$ is uniformly random, i.e., $\pi_b(a) = \pi_b(b) = 0.5$.
- Target policy $\pi$ is deterministic, i.e., $\pi(a) = 1$

The support condition is satisfied since $\pi_b(a) > 0$ when $\pi_b(b) > 0$.

In this case, the importance sampling ratio for a trajectory $\tau$ of length $T$ is:
$$
    W(\tau) = \prod_{t=0}^{T-1} \frac{\pi(a_t)}{\pi_b(a_t)} = \prod_{t=0}^{T-1} \frac{1}{0.5} = 2^T
$$

Next, let us calculate the mean:
$$
    \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ W(\tau) R(\tau) \right] = \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ 2^T T \right] = 2^T T \cdot 0.5^T = T
$$

The second moment is:
$$
    \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} [ \left( W(\tau) R(\tau) \right)^2 ] = \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ (2^T T)^2 \right] = 4^T T^2 \cdot 0.5^T = 2^T T^2
$$

Thus, the variance is:
$$
    \text{Var}\left( W(\tau) R(\tau) \right) = 2^T T^2 - T^2 = T^2 (2^T - 1)
$$
which grows exponentially with the trajectory length $T$.

---

Now, to prove the last part of the theorem, we need to show that if there exists $\epsilon > 0$ such that for every reachable state $s$, the per-step importance weight has second moment bounded below by $1 + \epsilon$, then the variance of the importance sampling estimator grows at least exponentially with the trajectory length $T$.

To analyze the growth of variance with trajectory length $T$, let us focus at the term $\mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} [ \left( W(\tau) R(\tau) \right)^2 ]$. 

The second term $\left( \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ W(\tau) R(\tau) \right] \right)^2 = v_{\pi}^{(T)}(s_0)^2$ is the squared expected return. Under bounded rewards, this term is finite and so bounded. Let $R_{max}$ be the maximum possible return, then 
$$
    \left| R(\tau) \right| \leq \sum_{t=0}^{T-1} \left| r_{t+1} \right| \leq \sum_{t=0}^{T-1} R_{max} = T R_{max}
$$
and so, 
$$
    \left( W(\tau) R(\tau) \right)^2 \leq W(\tau)^2 (T R_{max})^2 = T^2 R_{max}^2 W(\tau)^2
$$
which means that the variance is upper bounded by a term that grows quadratically with $T$ and not exponentially. 

Now, for the first term, we have
$$
    \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} [ \left( W(\tau) R(\tau) \right)^2 ] 
$$
and from the definition of the importance sampling ratio
$$
W(\tau)^2 = \prod_{t=0}^{T-1} \left( \frac{\pi(a_t | s_t)}{\pi_b(a_t | s_t)} \right)^2
$$
we have
$$
    \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} [ W(\tau)^2 ] = \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} \left[ \prod_{t=0}^{T-1} \left( \frac{\pi(a_t | s_t)}{\pi_b(a_t | s_t)} \right)^2 \right]
$$

For a single time step, let's compute the expected value of the squared importance weight:
$$\begin{aligned}
    \mathbb{E}_{a_t \sim \pi_b(\cdot | s_t)} \left[ \left( \frac{\pi(a_t | s_t)}{\pi_b(a_t | s_t)} \right)^2 \right] &= \sum_{a_t} \pi_b(a_t | s_t) \left( \frac{\pi(a_t | s_t)}{\pi_b(a_t | s_t)} \right)^2 \\
    &= \sum_{a_t} \frac{\pi(a_t | s_t)^2}{\pi_b(a_t | s_t)} \\
    &\geq \sum_{a_t} \pi(a_t | s_t) = 1
\   \end{aligned}$$
with equality if and only if $\pi(a_t | s_t) = \pi_b(a_t | s_t)$ for all actions $a_t$.

Note that $\sum_a \frac{\pi(a | s)^2}{\pi_b(a | s)} \geq \left( \sum_a \pi(a | s) \right)^2$ comes from Jensen's inequality applied to the convex function $f(x) = \frac{1}{x}$, which states that for any set of non-negative weights $w_i$ and corresponding values $x_i$, we have:
$$    
    f\left( \sum_i w_i x_i \right) \leq \sum_i w_i f(x_i) = \sum_i w_i \frac{1}{x_i}
$$
or in terms of expectations, if $X$ is a random variable and $f = \frac{1}{x}$ is convex, then:
$$
    f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)] = \mathbb{E}\left[ \frac{1}{X} \right]
$$

In our case, we can set $w_i = \pi(a_i | s)$ and $x_i = \frac{\pi_b(a_i | s)}{\pi(a_i | s)}$. Then, we have:
$$
    f \left( \sum_i w_i x_i \right) = f\left( \sum_i \pi(a_i | s) \frac{\pi_b(a_i | s)}{\pi(a_i | s)} \right) = f\left( \sum_i \pi_b(a_i | s) \right) = f(1) = 1
$$
and 
$$
    \sum_i w_i f(x_i) = \sum_i \pi(a_i | s) \frac{1}{\frac{\pi_b(a_i | s)}{\pi(a_i | s)}} = \sum_i \frac{\pi(a_i | s)^2}{\pi_b(a_i | s)}
$$
And, thus,
$$
    \sum_i \frac{\pi(a_i | s)^2}{\pi_b(a_i | s)} \geq 1
$$

Now, considering the entire trajectory, let $p = \min_{s} \mathbb{E}_{a \sim \pi_b(\cdot|s)}\left[ \left(\frac{\pi(a|s)}{\pi_b(a|s)}\right)^2 \right]$. By the hypothesis of the theorem, $p \geq 1 + \epsilon > 1$. We want to show that $\mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} [ W(\tau)^2 ] \geq p^T$.

In a general MDP, the per-step weights $w_0, \ldots, w_{T-1}$ are not independent across time steps since the state $s_t$ depends on all previous actions. We therefore cannot factor the expectation directly. Instead, we apply the **tower property** (law of total expectation), conditioning on the state $s_{T-1}$ at the last step and peeling off one step at a time:
$$
    \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)}\! \left[ \prod_{t=0}^{T-1} w_t^2 \right] = \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)}\!\left[ \prod_{t=0}^{T-2} w_t^2 \cdot \mathbb{E}\!\left[ w_{T-1}^2 \,\Big|\, s_{T-1} \right] \right] \geq p \cdot \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)}\!\left[ \prod_{t=0}^{T-2} w_t^2 \right]
$$
since $\mathbb{E}[w_{T-1}^2 \mid s_{T-1}] \geq p$ for every reachable state $s_{T-1}$. Applying this argument inductively over all $T$ steps gives:
$$
    \mathbb{E}_{\tau \sim P(\cdot | s_0, \pi_b)} [ W(\tau)^2 ] \geq p^T
$$

To close the proof, we connect this to the variance. The second term $v_\pi(s_0)^2$ is bounded independently of $T$ since rewards are bounded. For the first term, since rewards are bounded away from zero ($R(\tau) \geq r_{\min} > 0$ for every trajectory), we have $W(\tau)^2 R(\tau)^2 \geq r_{\min}^2 W(\tau)^2$ pointwise, and therefore:
$$
    \mathbb{E}[W(\tau)^2 R(\tau)^2] \geq r_{\min}^2 \, \mathbb{E}[W(\tau)^2] \geq r_{\min}^2 \, p^T
$$
Thus:
$$
    \text{Var}\!\left( W(\tau) R(\tau) \right) = \mathbb{E}[W(\tau)^2 R(\tau)^2] - v_\pi(s_0)^2 \geq r_{\min}^2 \, p^T - v_\pi(s_0)^2
$$
which grows at least exponentially with $T$. $\qquad \qquad \qquad \square$

### Weighted Importance Sampling (WIS)

The estimator above uses ordinary importance sampling (OIS):
$$
\hat{v}_{\text{OIS}}(s_0) = \frac{1}{n} \sum_{i=1}^{n} W_i R_i,
$$
where $W_i = W(\tau_i)$ and $R_i = R(\tau_i)$ for trajectories sampled from $\pi_b$ and $n$ is the number of trajectories.

Weighted importance sampling keeps the same trajectory ratios, but normalizes by the sum of weights:
$$
\hat{v}_{\text{WIS}}(s_0) = \frac{\sum_{i=1}^{n} W_i R_i}{\sum_{i=1}^{n} W_i}.
$$
Equivalently, with normalized weights $\tilde{W}_i = \frac{W_i}{\sum_{j=1}^{n} W_j}$, we can write
$$
\hat{v}_{\text{WIS}}(s_0) = \sum_{i=1}^{n} \tilde{W}_i R_i,
$$
so the estimate is a weighted average of returns whose weights sum to $1$.

This directly addresses the variance problems discussed above. In OIS, a few trajectories with very large $W_i$ can dominate the estimate. In WIS, those trajectories still get large relative influence, but the global normalization prevents the overall scale from exploding as easily.

The tradeoff matches the earlier bias-variance discussion:
- OIS is unbiased under the support condition $\pi_b(a \mid s) > 0$ whenever $\pi(a \mid s) > 0$, but can have very high variance.
- WIS usually has lower variance, but is biased for finite $n$ because of the random denominator $\sum_i W_i$.

Suppose in OIS, one of the samples has an extremely large weight $W_k$ due to a rare trajectory that is much more likely under $\pi$ than $\pi_b$. This single sample can dominate the sum $\sum_i W_i R_i$, leading to an estimate that is highly sensitive to that one trajectory. In contrast, in WIS, while $W_k$ may still be large, the normalization by $\sum_i W_i$ ensures that the overall estimate does not become excessively large due to a single outlier. The estimate could be dominated by that trajectory, but it won't blow up the estimate to an extreme value. This makes WIS more stable and less sensitive to extreme weights, thus reducing variance.

As $n \to \infty$, $\hat{v}_{\text{WIS}}(s_0)$ is consistent and converges to $v_\pi(s_0)$ under standard regularity conditions. In practice, WIS is often preferred when stability is more important than exact finite-sample unbiasedness.

---

__Theorem 3 (Bias of Weighted Importance Sampling)__ : For a fixed sample size $N$, the self-normalized importance sampling estimator $\hat{v}_{\text{WIS}}(s_0)$ is biased for estimating $v_\pi(s_0)$, i.e., $\mathbb{E}[\hat{v}_{\text{WIS}}(s_0)] \neq v_\pi(s_0)$ in general. However, as $N \to \infty$, $\hat{v}_{\text{WIS}}(s_0)$ is consistent and converges to $v_\pi(s_0)$ under standard regularity conditions.

__Proof__: Write the WIS estimator as a ratio of sample averages:
$$
    \hat{v}_{\text{WIS}}(s_0) = \frac{\bar{A}}{\bar{B}}, \qquad \bar{A} = \frac{1}{N}\sum_{i=1}^{N} W_i R_i, \qquad \bar{B} = \frac{1}{N}\sum_{i=1}^{N} W_i
$$
To prove finite-sample bias formally, it is enough to give one counterexample where
$$
    \mathbb{E}\left[\hat{v}_{\text{WIS}}(s_0)\right] \neq v_\pi(s_0)
$$
for every fixed finite $N$.

Consider a one-step bandit MDP with two actions $a,b$ and rewards
$$
    r(a) = 1, \qquad r(b) = 0.
$$
Let the behavior and target policies be
$$
    \pi_b(a) = \pi_b(b) = \frac{1}{2}, \qquad
    \pi(a) = \frac{3}{4}, \quad \pi(b) = \frac{1}{4}.
$$
Then
$$
    v_\pi(s_0) = \mathbb{E}_{A \sim \pi}[r(A)] = \frac{3}{4}.
$$

For $i=1,\ldots,N$, sample $A_i \sim \pi_b$ i.i.d. and define
$$
    R_i = \mathbf{1}\{A_i=a\}, \qquad
    W_i = \frac{\pi(A_i)}{\pi_b(A_i)}
    =
    \begin{cases}
    \frac{3}{2}, & A_i=a, \\
    \frac{1}{2}, & A_i=b.
    \end{cases}
$$
Let
$$
    K = \sum_{i=1}^{N} \mathbf{1}\{A_i=a\} \sim \text{Binomial}\!\left(N,\frac{1}{2}\right).
$$
Then
$$
    \sum_{i=1}^{N} W_i R_i = \frac{3}{2}K,
    \qquad
    \sum_{i=1}^{N} W_i = \frac{3}{2}K + \frac{1}{2}(N-K) = \frac{N+2K}{2},
$$
so
$$
    \hat{v}_{\text{WIS}}(s_0) = \frac{3K}{N+2K} =: f(K).
$$

Now
$$
    f''(k) = -\frac{12N}{(N+2k)^3} < 0,
$$
so $f$ is strictly concave on $[0,N]$. Since $K$ is non-degenerate for any finite $N \ge 1$, strict Jensen gives
$$
    \mathbb{E}[f(K)] < f(\mathbb{E}[K]).
$$
Using $\mathbb{E}[K] = N/2$, we get
$$
    \mathbb{E}\left[\hat{v}_{\text{WIS}}(s_0)\right]
    = \mathbb{E}[f(K)]
    < f\!\left(\frac{N}{2}\right)
    = \frac{3}{4}
    = v_\pi(s_0).
$$
Hence, for this MDP and every fixed finite $N$, WIS is biased. Therefore, WIS is biased in general at finite sample size.

---

For consistency, since the trajectories $\tau_1, \ldots, \tau_N$ are drawn i.i.d. from $\pi_b$, the Strong Law of Large Numbers gives, almost surely as $N \to \infty$:
$$
\begin{aligned}
    \bar{A} &= \frac{1}{N}\sum_{i=1}^{N} W_i R_i \;\xrightarrow{\;a.s.\;}\; \mathbb{E}_{\pi_b}[W R] = v_\pi(s_0) \\
    \bar{B} &= \frac{1}{N}\sum_{i=1}^{N} W_i \;\xrightarrow{\;a.s.\;}\; \mathbb{E}_{\pi_b}[W] = 1
\end{aligned}
$$
under the regularity conditions $\mathbb{E}_{\pi_b}[|WR|] < \infty$ and $\mathbb{E}_{\pi_b}[W] < \infty$. Since $\bar{B} \to 1 \neq 0$ and division is continuous, the continuous mapping theorem gives:
$$
    \hat{v}_{\text{WIS}}(s_0) = \frac{\bar{A}}{\bar{B}} \;\xrightarrow{\;a.s.\;}\; \frac{v_\pi(s_0)}{1} = v_\pi(s_0)
$$
Thus $\hat{v}_{\text{WIS}}(s_0)$ is consistent. $\qquad \qquad \qquad \square$
