+++
title = 'Actor Critic Algorithms'
date = 2026-04-25T23:25:54-04:00
draft = false
+++

In this article, we will explore actor critic algorithms with respect to four different algorithms that illustrate the progression from pure policy gradients to parallelized advantage actor-critic methods. Each algorithm has its own unique properties that contribute in its own way to the stability and efficiency of learning. 

We start with **Vanilla REINFORCE**, which relies entirely on episodic Monte Carlo returns; **TD Actor-Critic**, which introduces a critic to bootstrap value estimates and enables step-by-step updates; **Actor-Critic with Advantage**, which computes episodic advantage using baseline value predictions to reduce gradient variance; and **A2C with Parallel Workers**, which scales the advantage method across vectorized environments for more stable and efficient learning.

## Vanilla REINFORCE

Vanilla REINFORCE is a pure policy gradient method that operates without a critic network. In this approach, the agent collects a full episode of experience and calculates the raw discounted return $G_t$ for each timestep. The actor's policy parameters are then updated in the direction of the log probability of the sampled actions, scaled directly by $G_t$:

$$ \theta \leftarrow \theta + \alpha \cdot G_t \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) $$

| Symbol | Meaning |
|---|---|
| $\theta$ | Policy parameters (weights of the actor network) |
| $\alpha$ | Learning rate |
| $G_t$ | Discounted return from timestep $t$ to the end of the episode |
| $\nabla_\theta$ | Gradient with respect to the parameters $\theta$ |
| $\pi_\theta(a_t\|s_t)$ | Probability of taking action $a_t$ in state $s_t$ given policy parameterized by $\theta$ |
| $a_t$ | Action taken at timestep $t$ |
| $s_t$ | State at timestep $t$ |

Because it relies on the actual Monte Carlo return over the entire episode, REINFORCE produces an unbiased estimate of the policy gradient. However, this full-trajectory reliance inherently suffers from high variance, as the total return can fluctuate significantly based on stochasticity in both the policy and the environment. This high variance typically leads to slow and unstable convergence.

### Pseudocode

```python
Initialize policy network parameters θ
for each episode:
    trajectory = []
    state = env.reset()
    while episode is not done:
        action = sample from policy π_θ(state)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
    
    for t, (state, action, reward) in enumerate(trajectory):
        # Calculate discounted return from step t
        G_t = compute_discounted_return(trajectory, t)
        
        # Update policy parameters
        θ = θ + α * G_t * ∇_θ log(π_θ(action|state))
```

## TD Actor-Critic

To address the high variance of Monte Carlo returns, the pure **TD (Temporal Difference) Actor-Critic** algorithm introduces a value-function approximator known as the critic. Instead of waiting until the end of the episode to compute $G_t$, the algorithm updates both the actor and the critic at every single timestep.

After taking an action and observing the immediate reward and next state, the critic computes the TD error $\delta$:

$$ \delta = r_{t+1} + \gamma V(s_{t+1}) - V(s_t) $$

| Symbol | Meaning |
|---|---|
| $\delta$ | TD (Temporal Difference) error, which serves as the advantage estimate |
| $r_{t+1}$ | Reward received after taking an action at timestep $t$ |
| $\gamma$ | Discount factor (between 0 and 1) |
| $V(s_{t+1})$ | Critic's estimated value of the next state $s_{t+1}$ |
| $V(s_t)$ | Critic's estimated value of the current state $s_t$ |
| $s_t$ | State at timestep $t$ |
| $s_{t+1}$ | State at timestep $t+1$ |

This TD error serves as an estimate of the advantage. The critic minimizes the squared TD error to improve its value estimates, while the actor uses $\delta$ to scale its policy gradient update. This leads to the following update equations for the critic parameters $w$ and actor parameters $\theta$:

$$ w \leftarrow w + \alpha_w \cdot \delta \cdot \nabla_w V_w(s_t) $$

$$ \theta \leftarrow \theta + \alpha_\theta \cdot \delta \cdot \nabla_\theta \log \pi_\theta(a_t|s_t) $$

| Symbol | Meaning |
|---|---|
| $w$ | Critic parameters (weights of the value network) |
| $\alpha_w$ | Learning rate for the critic |
| $\theta$ | Actor parameters (weights of the policy network) |
| $\alpha_\theta$ | Learning rate for the actor |
| $\nabla_w V_w(s_t)$ | Gradient of the value function with respect to critic parameters $w$ |
| $\nabla_\theta$ | Gradient with respect to actor parameters $\theta$ |

By bootstrapping—using current value estimates to predict future returns—TD Actor-Critic drastically reduces the variance of the updates, allowing the agent to learn faster and make online adjustments without waiting for an episode to terminate.

### Pseudocode

```python
Initialize actor network parameters θ
Initialize critic network parameters w
for each episode:
    state = env.reset()
    while episode is not done:
        action = sample from policy π_θ(state)
        next_state, reward, done = env.step(action)
        
        # Compute TD error
        target = reward + γ * V_w(next_state) * (1 - done)
        δ = target - V_w(state)
        
        # Update Critic (minimize TD error squared)
        w = w - α_w * ∇_w (target - V_w(state))²
        
        # Update Actor (policy gradient using TD error as advantage)
        θ = θ + α_θ * δ * ∇_θ log(π_θ(action|state))
        
        state = next_state
```

## Actor-Critic with Advantage (Episode-Level A2C)

The **Actor-Critic with Advantage** method bridges the gap between episodic returns and learned value baselines. Rather than relying on step-by-step TD errors, this algorithm waits until the end of an episode to perform updates. It computes the full discounted return $G_t$ for each step and compares it to the critic's baseline prediction $V(s_t)$. 

The difference between the actual return and the baseline is the advantage:

$$ A_t = G_t - V(s_t) $$

| Symbol | Meaning |
|---|---|
| $A_t$ | Advantage at timestep $t$ |
| $G_t$ | Actual discounted return from timestep $t$ to the end of the episode |
| $V(s_t)$ | Baseline prediction (Critic's estimated value) of state $s_t$ |
| $s_t$ | State at timestep $t$ |

The actor is then updated using this explicit advantage $A_t$ rather than the raw return $G_t$ or the one-step TD error. By subtracting the baseline value $V(s_t)$ from the target $G_t$, this method significantly reduces the variance of the policy gradient compared to Vanilla REINFORCE, while remaining an unbiased episodic estimate since it does not rely on the critic's bootstrapping to define the return target.

### Pseudocode

```python
Initialize actor network parameters θ
Initialize critic network parameters w
for each episode:
    trajectory = []
    state = env.reset()
    while episode is not done:
        action = sample from policy π_θ(state)
        next_state, reward, done = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        
    for t, (state, action, reward) in enumerate(trajectory):
        # Calculate full discounted return
        G_t = compute_discounted_return(trajectory, t)
        
        # Compute advantage
        A_t = G_t - V_w(state)
        
        # Update Critic (minimize error between return and baseline)
        w = w - α_w * ∇_w (G_t - V_w(state))²
        
        # Update Actor (policy gradient with advantage)
        θ = θ + α_θ * A_t * ∇_θ log(π_θ(action|state))
```

## A2C with Parallel Workers

The final algorithm, **A2C (Advantage Actor-Critic) with Parallel Workers**, extends the advantage formulation by leveraging vectorized environments. Instead of running a single agent through one environment at a time, this method runs multiple parallel worker environments synchronously. 

At each step, the global network gathers transitions (states, actions, rewards) across all parallel workers. This batch of diverse experiences is used to compute advantages and perform a unified gradient update. Parallelizing the data collection has two major benefits:
1. **Decorrelation**: The simultaneous experiences are drawn from different parts of the state space, decorrelating the batch data and breaking temporal dependencies without the need for a memory-intensive replay buffer.
2. **Efficiency**: Batched inference and vectorized environments allow the algorithm to maximally utilize modern hardware, speeding up the wall-clock time of training.

This synchronous batched approach retains the stability improvements of the original asynchronous method (A3C) while being computationally simpler and highly efficient.

### Pseudocode

```python
Initialize global actor network parameters θ
Initialize global critic network parameters w
Initialize N parallel environments

for each update_iteration:
    batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = [], [], [], [], []
    
    # Collect a batch of experiences from all parallel environments synchronously
    states = envs.get_current_states()
    for step in range(steps_per_update):
        actions = sample from policy π_θ(states)
        next_states, rewards, dones = envs.step(actions)
        
        batch_states.append(states)
        batch_actions.append(actions)
        batch_rewards.append(rewards)
        batch_next_states.append(next_states)
        batch_dones.append(dones)
        
        states = next_states
        
    # Compute advantages and targets for the gathered batch
    batch_advantages = []
    batch_targets = []
    for states, rewards, next_states, dones in zip(batch_states, batch_rewards, batch_next_states, batch_dones):
        targets = rewards + γ * V_w(next_states) * (1 - dones)
        advantages = targets - V_w(states)
        
        batch_targets.append(targets)
        batch_advantages.append(advantages)
        
    # Perform unified gradient updates on the global networks
    w = w - α_w * mean( ∇_w (batch_targets - V_w(batch_states))² )
    θ = θ + α_θ * mean( batch_advantages * ∇_θ log(π_θ(batch_actions|batch_states)) )
```

These algorithms are implemented in [actor-critic cartpole](https://mochan.dev/actorcritic/), where we can train a CartPole agent using the above algorithms to compare their performance, view the training results, and examine the policy and value functions using neural networks and linear function approximators.