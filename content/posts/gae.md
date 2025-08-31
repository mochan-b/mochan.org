+++
title = 'GAE: Generalized Advantage Estimation'
author = 'Mochan Shrestha'
date = 2025-08-31T12:00:00-04:00
draft = false
+++

In reinforcement learning, agents learn to make better decisions by understanding not just which actions are good, but how much better they are than what the agent would typically expect in each situation. This comparison is captured by the advantage function, which measures how much better (or worse) taking a specific action is compared to the expected value of simply being in that state.

However, estimating these advantages accurately presents a fundamental challenge that lies at the heart of many policy gradient methods. We face a classic bias-variance tradeoff: we can estimate advantages using methods that are mathematically correct on average (low bias) but have high variability between samples (high variance), or we can use approximations that are more stable (low variance) but systematically over- or under-estimate the true advantages (high bias).

High variance makes learning unstable and sample-inefficient—the agent's understanding of which actions are good fluctuates wildly between episodes. High bias, on the other hand, can lead the agent to consistently misunderstand the value of its actions, potentially converging to suboptimal policies.

Generalized Advantage Estimation (GAE) provides an solution to this dilemma by offering a spectrum of estimators that smoothly interpolate between high-bias, low-variance estimates and low-bias, high-variance ones. This allows the user to tune the bias-variance tradeoff based on their specific problem's characteristics, leading to more stable and efficient policy learning.

GAE achieves this by using a hyperparameter λ (lambda) that exponentially weights different n-step advantage estimates—when λ is close to 0, it emphasizes short-term, stable estimates (high bias, low variance), and when λ approaches 1, it incorporates longer-term trajectory information (low bias, high variance), creating a continuous spectrum between these extremes.

---

## Advantage Function Recap

The advantage function is defined as:

$$
A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)
$$

where:

* $Q^\pi(s_t, a_t)$ is the expected return after taking action $a_t$ at state $s_t$.
* $V^\pi(s_t)$ is the expected return from state $s_t$ under policy $\pi$.

Intuitively, $A^\pi(s_t, a_t)$ tells us how much better an action is compared to the average behavior of the policy at that state.

## Estimating Advantages: Monte Carlo vs Bootstrapping

There are two common ways to estimate the return $Q^\pi(s_t, a_t)$ in the advantage formula:

1. **Monte Carlo Returns**

   * Compute the full discounted sum of rewards until the end of an episode:

     $$
     G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k}
     $$
   * Plug this into the advantage: $A_t = G_t - V(s_t)$.
     - _Strength_: unbiased (on average, exactly correct).
     - _Weakness_: very high variance because it depends on the full trajectory.

2. **Short-Horizon Bootstrapped Returns**

   * Use value function estimates to approximate future rewards after a few steps:

     $$
     G_t^{(n)} = r_t + \gamma r_{t+1} + \dots + \gamma^{n-1} r_{t+n-1} + \gamma^n V(s_{t+n})
     $$
   * Advantage: $A_t^{(n)} = G_t^{(n)} - V(s_t)$.
     - _Strength_: lower variance since it relies less on full trajectories.
     - _Weakness_: biased, since the value function may not be perfect.

These two approaches map directly back to the advantage function formula: Monte Carlo returns give an unbiased but noisy estimate of $Q^\pi$, while short-horizon bootstrapped returns give a biased but more stable estimate.

## The Bias–Variance Tradeoff

Estimating advantages accurately presents a fundamental challenge:

* **High variance** occurs with Monte Carlo returns. Estimates are correct on average but fluctuate wildly between episodes, making learning unstable and sample-inefficient.
* **High bias** occurs with short-horizon bootstrapped returns. Estimates are more stable but systematically misestimate the true advantages, leading to convergence toward suboptimal policies.

---

## Motivation

In policy gradient methods, the policy update depends on the gradient of the expected return. Practically, this gradient involves the advantage function. The key challenge is reducing variance without introducing too much bias.

**Generalized Advantage Estimation (GAE)** provides a solution to this dilemma by offering a spectrum of estimators that smoothly interpolate between high-bias, low-variance estimates and low-bias, high-variance ones. This allows the practitioner to tune the bias–variance tradeoff based on the problem’s characteristics, leading to more stable and efficient policy learning.

---

## Temporal-Difference Residual (TD Error)

The one-step TD error is defined as:

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

This quantity measures the immediate correction to the value estimate. It is the building block of GAE.

## Generalized Advantage Estimation

Instead of relying on only one-step or full Monte Carlo returns, GAE defines the advantage estimate as an exponentially-weighted average of multi-step TD errors:

$$
A_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
$$

where:

* $\gamma$ is the discount factor.

* $\lambda \in [0,1]$ is the GAE parameter that controls the bias–variance tradeoff.

* **When ************$\lambda = 1$************:** GAE reduces to Monte Carlo estimation (low bias, high variance).

* **When ************$\lambda = 0$************:** GAE reduces to one-step TD estimation (low variance, higher bias).

* **Intermediate ************$\lambda$************:** Smooth tradeoff between the two extremes.

---

## Algorithm

The practical implementation of GAE is typically done in reverse order through a trajectory:

1. Initialize $A_t = 0$.
2. Iterate backward through the trajectory:

   $$
   A_t = \delta_t + \gamma \lambda (1 - d_t) A_{t+1}
   $$

   where $d_t$ indicates whether the episode ended.
3. Compute the returns as:

   $$
   R_t = A_t + V(s_t)
   $$

This recursive computation is efficient and widely used in PPO, TRPO, and many other modern RL algorithms.

### PyTorch Implementation

Below is a concise and production-ready PyTorch implementation for batched environments. It expects tensors shaped as **\[T, B]** (time-major) for `rewards`, `dones`, and **\[T+1, B]** for `values` (note the bootstrap value at `t = T`).

```python
import torch

def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """
    Generalized Advantage Estimation (GAE).

    Args:
        rewards: Tensor [T, B]
        values:  Tensor [T+1, B]  (bootstrap value in the last row)
        dones:   Tensor [T, B]     (1.0 if episode ended at step t, else 0.0)
        gamma:   discount factor (float)
        lam:     GAE lambda (float)

    Returns:
        advantages: Tensor [T, B]
        returns:    Tensor [T, B]  where returns = advantages + values[:-1]
    """
    # Validate input shapes where T is time steps and B is batch size
    T, B = rewards.shape
    assert values.shape == (T + 1, B), "values must be time-major with T+1 rows for bootstrapping"

    # Initialize tensors
    advantages = torch.zeros_like(rewards) # [T, B]. Holds the advantage estimates
    gae = torch.zeros(B, dtype=values.dtype, device=values.device) # [B]. Holds the running GAE value

    # Compute GAE in reverse order from T-1 down to 0 of the trajectory
    for t in reversed(range(T)):
        # If the episode ended at step t, we do not propagate the advantage
        not_done = 1.0 - dones[t]
        # TD error
        delta = rewards[t] + gamma * values[t + 1] * not_done - values[t]
        # GAE recursive formula
        gae = delta + gamma * lam * not_done * gae
        # Store the advantage estimate
        advantages[t] = gae

    # Compute returns
    returns = advantages + values[:-1]
    return advantages, returns

# Example usage:
# T, B = 128, 16
# rewards = torch.randn(T, B)
# dones   = torch.zeros(T, B)  # set to 1.0 where episodes end
# values  = torch.randn(T + 1, B)
# adv, ret = compute_gae(rewards, values, dones, gamma=0.99, lam=0.95)
```

The rewards tensor contains the immediate rewards you observed at each timestep. The values tensor contains the value function estimates for each state, with the last row being the bootstrap value after the final timestep. The dones tensor indicates whether an episode ended at each timestep, which is crucial for correctly handling episode boundaries.

First, we compute the TD error (delta). This is the one-step advantage estimate—it tells us how much better or worse this timestep was compared to what our value function expected. Second, we update our running GAE value. This is accumulating a weighted sum of all future TD errors. 

---

## Conclusion

Generalized Advantage Estimation (GAE) is a cornerstone of modern policy gradient methods. By blending TD and Monte Carlo returns with a tunable parameter $\lambda$, it provides a practical way to reduce variance and stabilize learning in reinforcement learning. Its widespread adoption in algorithms like PPO highlights its effectiveness in balancing the exploration-exploitation tradeoff and enabling scalable RL training.

---

**References:**

* Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2015). *High-Dimensional Continuous Control Using Generalized Advantage Estimation*. arXiv:1506.02438. https://arxiv.org/abs/1506.02438v1
