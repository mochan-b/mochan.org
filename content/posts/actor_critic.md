+++
title = 'Actor Critic Methods in Reinforcement Learning'
date = 2025-03-21T08:05:36-04:00
draft = false
+++

Actor-critic methods combine value-based learning and policy-based learning, using two separate functions: the **critic**, which learns the value function, and the **actor**, which learns the policy. They work together. The critic evaluates the actions taken by the actor and provides feedback to improve the policy.

## Value-Based Learning

In value-based learning, the agent learns to estimate the **value** of being in a given state. That is, the expected cumulative reward the agent can obtain from that state onward. The most common value functions are:

- **V(s)**:  the state-value function: the expected return starting from state *s* and following a given policy.
- **Q(s, a)**: the action-value function: the expected return after taking action *a* in state *s* and then following a given policy.

The agent uses these estimates to derive a policy, typically by acting greedily, choosing the action with the highest estimated Q-value. Well-known value-based algorithms include Q-learning and DQN. However, value-based methods struggle with continuous action spaces and do not directly learn a policy, making them less suitable for tasks that require fine-grained action control.

## Policy-Based Learning

In policy-based learning, the agent directly learns a **policy** $π(a|s)$: a mapping from states to actions (or a probability distribution over actions) — without explicitly estimating value functions. The policy is typically parameterised by a neural network with weights θ, and training proceeds by optimizing these weights to maximize expected cumulative reward.

The core idea is to compute the **policy gradient**: the gradient of expected return with respect to θ. This gradient indicates how to adjust the policy parameters to make higher-reward actions more probable. The REINFORCE algorithm is the canonical policy gradient method, updating the policy using:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot G_t \right]$$

where $G_t$ is the return from timestep $t$. Policy-based methods handle continuous action spaces naturally and can learn stochastic policies, but they tend to suffer from high variance in gradient estimates and slow convergence.

## Combining Both: Actor-Critic

Actor-critic methods address the weaknesses of each approach by combining them. The **actor** is a policy network $π(a|s; θ)$ that selects actions, inheriting the ability to handle continuous actions and learn stochastic policies from policy-based learning. The **critic** is a value network $V(s; w)$ or $Q(s, a; w)$ that evaluates the actor's choices — providing a low-variance learning signal in place of the noisy Monte Carlo returns used in REINFORCE.

Instead of using the full return $G_t$ to scale the policy gradient, the actor uses the critic's estimate as feedback. A common formulation replaces $G_t$ with the **advantage** $A(s, a) = Q(s, a) - V(s)$, which measures how much better an action is compared to the average. This reduces variance while keeping the gradient estimate unbiased:

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi \left[ \nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a) \right]$$

The two networks are trained simultaneously: the critic minimizes the error in its value estimates (e.g. via TD learning), while the actor updates its policy using the advantage signal provided by the critic. This interplay allows actor-critic methods to be more stable and sample-efficient than pure policy gradient methods.

## The Actor-Critic Algorithm

At each timestep, the agent observes a state, the actor selects an action, the environment returns a reward and next state, and both networks are updated. The critic is updated first using the **TD error** δ, which serves as the advantage estimate. Here $s'$ denotes the next state observed after taking action $a$ in state $s$:

$$\delta = r + \gamma V(s'; w) - V(s; w)$$

This measures how much better the outcome was compared to the critic's prediction. A positive δ means the action led to a better-than-expected result; a negative δ means it was worse. The actor then uses this signal to reinforce or discourage the action taken.

We use the TD error $\delta$ to update both the critic and the actor. The critic parameters *w* are updated to minimize the squared TD error, while the actor parameters θ are updated along the policy gradient, scaled by δ. The learning rates α_θ and α_w control the step sizes for the actor and critic updates, respectively.

The critic parameters *w* are updated by minimizing the squared TD error:

$$w \leftarrow w + \alpha_w \cdot \delta \cdot \nabla_w V(s; w)$$

The actor parameters θ are updated along the policy gradient, scaled by δ:

$$\theta \leftarrow \theta + \alpha_\theta \cdot \delta \cdot \nabla_\theta \log \pi_\theta(a|s)$$

**Pseudocode:**

```
Initialise actor parameters θ and critic parameters w arbitrarily
Set discount factor γ, learning rates α_θ and α_w

For each episode:
    Observe initial state s

    For each timestep t:
        Sample action a ~ π(a|s; θ)
        Take action a, observe reward r and next state s'

        # Compute TD error (advantage estimate)
        δ = r + γ * V(s'; w) - V(s; w)

        # Update critic
        w ← w + α_w * δ * ∇_w V(s; w)

        # Update actor
        θ ← θ + α_θ * δ * ∇_θ log π_θ(a|s)

        s ← s'

        If s' is terminal: break
```

## Variations of Actor-Critic

The vanilla actor-critic described above, while foundational, has practical limitations: training can be unstable, updates are correlated when generated from a single environment, and the critic can overestimate values. A number of algorithms have been developed to address these issues:

- **A3C (Asynchronous Advantage Actor-Critic)** — runs multiple independent agents in parallel, each interacting with its own copy of the environment and computing gradients asynchronously. This decorrelates experience and improves stability without the need for a replay buffer.
- **A2C (Advantage Actor-Critic)** — a synchronous variant of A3C that waits for all parallel workers to finish before performing a single, batched update. This is simpler to implement and often performs comparably to A3C.
- **TRPO (Trust Region Policy Optimization)** — constrains each policy update to stay within a "trust region", preventing large destabilising steps. It does this by enforcing a hard KL-divergence constraint between the old and new policy.
- **PPO (Proximal Policy Optimization)** — achieves a similar effect to TRPO but replaces the hard constraint with a clipped surrogate objective, making it simpler and more computationally efficient. PPO is one of the most widely used actor-critic algorithms in practice.
- **DDPG (Deep Deterministic Policy Gradient)** — extends actor-critic to deterministic policies for continuous action spaces, using a replay buffer and target networks borrowed from DQN to stabilise training.
- **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** — builds on DDPG by using two critic networks and taking the minimum of their estimates to reduce overestimation bias. It also delays actor updates relative to critic updates to improve stability.
- **SAC (Soft Actor-Critic)** — introduces an entropy regularisation term into the objective, encouraging the policy to remain as random as possible while still maximising reward. This improves exploration and robustness.

| Algorithm | Parallel Workers | Replay Buffer | Policy Constraint / Clipping | Twin Critics | Entropy Regularisation | Deterministic Policy | Key Problem Addressed |
|-----------|:----------------:|:-------------:|:----------------------------:|:------------:|:----------------------:|:--------------------:|----------------------|
| Vanilla AC | No | No | No | No | No | No | Baseline |
| A3C | Yes (async) | No | No | No | No | No | Correlated updates, slow training |
| A2C | Yes (sync) | No | No | No | No | No | Correlated updates, simpler than A3C |
| TRPO | No | No | Hard KL constraint | No | No | No | Unstable, destructively large updates |
| PPO | No | No | Clipped objective | No | No | No | Unstable updates (simpler than TRPO) |
| DDPG | No | Yes | No | No | No | Yes | Continuous action spaces |
| TD3 | No | Yes | No | Yes | No | Yes | Critic overestimation bias |
| SAC | No | Yes | No | Yes | Yes | No | Poor exploration, sample inefficiency |

## Appendix: Symbol Reference

| Symbol | Meaning |
|--------|---------|
| $s$ | Current state |
| $s'$ | Next state (observed after taking action $a$ in state $s$) |
| $a$ | Action taken by the actor |
| $r$ | Reward received from the environment |
| $t$ | Current timestep |
| $\gamma$ | Discount factor (controls how much future rewards are valued) |
| $\pi(a \| s)$ | Policy: probability of taking action $a$ in state $s$ |
| $\pi_\theta(a \| s)$ | Policy parameterised by $\theta$ |
| $\theta$ | Actor network parameters |
| $V(s)$ | State-value function: expected return from state $s$ |
| $V(s; w)$ | State-value function parameterised by $w$ |
| $Q(s, a)$ | Action-value function: expected return after taking action $a$ in state $s$ |
| $Q(s, a; w)$ | Action-value function parameterised by $w$ |
| $w$ | Critic network parameters |
| $A(s, a)$ | Advantage function: $Q(s, a) - V(s)$ |
| $\delta$ | TD error: $r + \gamma V(s'; w) - V(s; w)$ |
| $G_t$ | Return from timestep $t$: cumulative discounted reward |
| $J(\theta)$ | Expected return under policy $\pi_\theta$ |
| $\alpha_\theta$ | Learning rate for the actor |
| $\alpha_w$ | Learning rate for the critic |

