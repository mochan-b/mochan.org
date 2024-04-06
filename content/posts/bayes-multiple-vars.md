+++
title = 'Bayes Theorem Conditioned on Multiple Variables'
author = 'Mochan Shrestha'
date = 2024-04-06T15:08:01-04:00
draft = false
+++

We want to prove Bayes theorem that is conditioned on multiple variables. We want to prove the following formula.
$$
P(A \mid B, C) = \frac{P(B \mid A,C) \cdot P(A \mid C)}{P(B \mid C)}
$$

First, we have the following identities that are true that we will use for proving the above formula.

From the definition of _conditional probability_, we have
$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

From _Bayes theorem_, we have
$$
P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}
$$

#### Proof:

Using the definition of conditional probability, we have
$$
\begin{align}
P(A|B,C) &= \frac{P(A \cap B \mid C) }{P(B \mid C)} \\
\end{align}
$$
Again using the definition of conditional probability, we have
$$
P(A \cap B \mid C) = \frac{P(A \cap B \cap C)}{P(C)}
$$
Adding to the above equation, 
$$
\begin{align}
P(A|B,C) &= \frac{P(A \cap B \mid C) }{P(B \mid C)} \\\\
&= \frac{P(A \cap B \cap C)}{P(C) \\; P(B \mid C)} \\\\
&= \frac{P(A \cap B \mid C)}{P(B \mid C)} \\\\
&= \frac{P(A \cap B \mid C)}{P(A \mid C)} \frac{P(A \mid C)}{P(B \mid C)} \\\\
&= \frac{P(B \mid A, C) \\; P(A \mid C)}{P(B \mid C)} \qquad \qquad \text{□} \\\\
\end{align}
$$
                                                                  