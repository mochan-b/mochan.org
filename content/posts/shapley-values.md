+++
title = 'Shapley Values'
author = 'Mochan Shrestha'
date = 2023-11-12T05:38:39-05:00
draft = false
+++

Given a group of players working cooperatively, Shapley value is a measure of the contribution of a participant in a game. Shapley values comes from game theory. This is known as the credit assignment problem; given the outcome how much of the credit belongs to which player.

Another way to interpret this is that if a group of players cooperated towards some payoff, what is the fair way to split the payoff depending on the contributions of each player.

Since the contribution of each player is not independent of the other players's contributions, it is not solely based on the individual but what they add or take from the cooperative game. If there is synergistic cooperation, they add more value than their individual value but if there is redundancy or antagonistic cooperation, then they subtract value.

In machine learning, Shapley values are used for feature attribution and importance, how much does a feature contribute to a prediction.

![Shapley Values](/images/shapley_values.png)

### Definition

Let $S$ be the set of players in the game of size $n$, i.e. $\mathcal{S} = n$. We want to find the Shapley value of player $i$.

Consider $\mathcal{S}$ all possible subsets of $S$ that do not include a player $i$.

Given a subset $\hat{S} \in \mathcal{S}$ , a subset of players that does not include $i$, we calculate the marginal contribution of player $i$ to $\hat{S}$. 

To calculate the marginal contribution, we define a value function $v:\mathcal{S} \rightarrow \mathbb{R}$ which takes a subset of $S$ and gives a real number value that the group of players can provide. 

Thus, the marginal contribution will just be
$$
v(\hat{S} \cup \\{i\\}) - v(\hat{S})
$$

To calculate the Shapley value, we average across all the of the subsets of $S$ with and without player $i$.

Since Shapley values also are affected by the order of the players and not just the players in the subset, we have to average across not just all the subsets but also across all of the orderings of a subset. The order in the subset can be thought of as when player joined and thus can affect the game.

One additional caveat is that the Shapley values of a player does not change when other players enter. So, for any Shapley value, only the value when $i$ enter matters and not afterwards when other players join. This reflects on how the averages are calculated in the formula below.

Let $R$ be a permutation of $S$ (a subset of $S$ where the ordering also matters). Let $\hat{R}$ be the subset of players that precede player $i$ in the permutation. Then, we can define the Shapley value of player $i$ with value function $v$ as
$$
\phi_i(v) = \frac{1}{n!} \sum_\hat{R} \left[ v(\hat{R} \cup \\{i\\}) - v(\hat{R}) \right]
$$

#### Alternate Formulation

There is also a slightly different formula based on subsets rather than permutation of $S$. Given our $\hat{S}$ a subset of $S$ that does not contain player $i$, 
$$
\frac{|\hat{S}|! (n - |\hat{S}| - 1)!}{n!}
$$gives the number of ways to select a players in $\hat{S}$ and players not in $\hat{S}$ excluding $i$. Then, we can give the Shapley value in terms of $\hat{S}$ as
$$
\phi_i(v) = \sum_{\hat{S}} \frac{|\hat{S}|! (n - |\hat{S}| - 1)!}{n!} (v(\hat{S} \cup \\{i\\}) - v(\hat{S}))
$$

## Properties

### Efficiency

The sum of the Shapley values of all the players is the total outcome from the game. 
$$
\sum_{i \in S} \phi_i(v) = v(S)
$$

#### Proof

$$
\begin{split}
\sum_{i \in S} \phi_i(v) &= \frac{1}{n!} \sum_{i \in S}  \sum_\hat{R} \left[ v(\hat{R} \cup \\{i\\}) - v(\hat{R}) \right] \\\
&= \frac{1}{n!} \sum_\hat{R} \sum_{i \in S} \left[ v(\hat{R} \cup \\{i\\}) - v(\hat{R}) \right] \\
\end{split}
$$

From our defintion, $\hat{R}$ is a permutation of $S$. Let the permutation be $(s_1, s_2, \ldots, s_n)$ where $s_1$ is the first player to join and $s_n$ is the last player to join.

$$
\begin{split}
\sum_{i \in S} \left[ v(\hat{R} \cup \\{i\\}) - v(\hat{R}) \right] &= [ v(S) - v(S - \\{s_n\\})] + [ v(S - \\{s_n\\}) - v(S - \\{s_n, s_{n-1}\\})] + \\\ 
&\ldots + [ v(S - \\{s_n, s_{n-1}, \ldots, s_2\\}) - v(S - \\{s_n, s_{n-1}, \ldots, s_1\\})] \\\
&= v(S) - v(\emptyset) \\\
&= v(S)
\end{split}
$$

### Symmetry

If two players have the same marginal contribution to every subset, then they have the same Shapley value.

Thus, for every $S \subset \mathcal{S} \backslash \\{i, j \\} $, if $v(S \cup \\{i\\}) - v(S) = v(S \cup \\{j\\}) - v(S)$, then $\phi_i(v) = \phi_j(v)$.

### Dummy Player

If a player has no marginal contribution to any subset, then their Shapley value is 0.

Thus, for every $S \subset \mathcal{S} \backslash \\{i\\} $, if $v(S \cup \\{i\\}) - v(S) = 0$, then $\phi_i(v) = 0$.

### Linearity

If we have two games $v_1$ and $v_2$, then the Shapley value of the sum of the games is the sum of the Shapley values of the games.

Thus, $\phi_i(v_1 + v_2) = \phi_i(v_1) + \phi_i(v_2)$.