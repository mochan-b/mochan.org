+++
title = 'Mixed Integer Linear Programming'
author = 'Mochan Shrestha'
date = 2023-11-25T08:47:30-05:00
draft = false
+++

Mixed integer linear programming is a set of optimization problems where some of the variables are constrained to be integer values. The problems are formulated as linear programs with additional constraints on some or all of the variables being integers. The problems are NP-hard in general, but are efficiently solvable in practice. The problems are also known as mixed integer programs (MIP), integer linear programs (ILP), or mixed integer linear programs (MILP).

In this article, we will look at a very basic example of mixed linear programming, find the solution using PuLP and discuss the branch and bound algorithm that is used to solve it.

## Example Problem

Suppose a company wants to produce 2 types of products, product A and product B. 

We have the following variables:
- $x_A$: number of units of product A to produce
- $x_B$: number of units of product B to produce
- $y$: additional resources to be used

We have the following objective function to maximize the profit:
$$
\begin{align}
\text{maximize profit} \quad & 15 x_A + 25 x_B - 10y \\
\end{align}
$$

The company has the following constraints:

$$
\begin{align}
\text{constraint 1} \quad & 3 x_A + 3.5 x_B - 10 y \leq 60 \\\
\text{constraint 2} \quad & 2 x_A + 3 x_B + y = 70 \\\
\text{min products constraint} \quad & x_A \geq 1 \\\
\text{max products constraint} \quad & x_A \leq 25 \\\
\text{min products constraint} \quad & x_B \geq 5 \\\
\text{resource constraint} \quad & y \geq 1 \\\
\end{align}
$$

## Solution using PuLP

We can solve this problem using PuLP. 
    
```python
from pulp import LpMaximize, LpProblem, LpVariable, LpStatus, PULP_CBC_CMD

# Define the problem
problem = LpProblem("Maximize_Profit", LpMaximize)

# Define the variables
x_A = LpVariable('x_A', lowBound=0, cat='Integer')  # Product A
x_B = LpVariable('x_B', lowBound=0, cat='Integer')  # Product B
y = LpVariable('y', lowBound=0)  # Continuous variable for Resource R2

# Objective function
problem += 15 * x_A + 25 * x_B - 10 * y, "Total Profit"

# Constraints
problem += 3 * x_A + 3.5 * x_B - 10 * y <= 60, "Resource R1 Constraint"
problem += 2 * x_A + 3 * x_B + y == 70, "Resource R2 Constraint"
problem += x_A + x_B >= 30, "Min products constraint"
problem += x_A >= 1, "Min product 1 constraint"
problem += x_A <= 25, "Max product 1 constraint"
problem += x_B >= 5, "Min product 2 constraint"
problem += y >= 1, "Min additional resource constraint"

# Solve the problem
problem.solve()

# Output the results
solution = {
    "Product A": x_A.varValue,
    "Product B": x_B.varValue,
    "Additional Resource R2": y.varValue,
    "Total Profit": problem.objective.value()
}

print(solution)
```

The output looks like the following::

```
Problem MODEL has 7 rows, 3 columns and 12 elements
Coin0008I MODEL read with 0 errors
Option for timeMode changed from cpu to elapsed
Continuous objective value is 483.333 - 0.00 seconds
Cgl0003I 0 fixed, 6 tightened bounds, 0 strengthened rows, 0 substitutions
Cgl0004I processed model has 3 rows, 3 columns (2 integer (0 of which binary)) and 8 elements
Cutoff increment increased from 1e-05 to 4.9999
Cbc0012I Integer solution of -470 found by DiveCoefficient after 0 iterations and 0 nodes (0.00 seconds)
Cbc0006I The LP relaxation is infeasible or too expensive
Cbc0013I At root node, 0 cuts changed objective from -483.33333 to -483.33333 in 1 passes
Cbc0014I Cut generator 0 (Probing) - 0 row cuts average 0.0 elements, 2 column cuts (2 active)  in 0.000 seconds - new frequency is 1
Cbc0014I Cut generator 1 (Gomory) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0014I Cut generator 2 (Knapsack) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0014I Cut generator 3 (Clique) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0014I Cut generator 4 (MixedIntegerRounding2) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0014I Cut generator 5 (FlowCover) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0014I Cut generator 6 (TwoMirCuts) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0014I Cut generator 7 (ZeroHalf) - 0 row cuts average 0.0 elements, 0 column cuts (0 active)  in 0.000 seconds - new frequency is -100
Cbc0001I Search completed - best objective -470, took 0 iterations and 0 nodes (0.00 seconds)
Cbc0035I Maximum depth 0, 0 variables fixed on reduced cost
Cuts at root node changed objective from -483.333 to -483.333
Probing was tried 1 times and created 2 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Gomory was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Knapsack was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
Clique was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
MixedIntegerRounding2 was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
FlowCover was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
TwoMirCuts was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)
ZeroHalf was tried 0 times and created 0 cuts of which 0 were active after adding rounds of cuts (0.000 seconds)

Result - Optimal solution found

Objective value:                470.00000000
Enumerated nodes:               0
Total iterations:               0
Time (CPU seconds):             0.00
Time (Wallclock seconds):       0.00

Option for printingOptions changed from normal to all
Total time (CPU seconds):       0.00   (Wallclock seconds):       0.00

{'Product A': 24.0, 'Product B': 6.0, 'Additional Resource R2': 4.0, 'Total Profit': 470.0}
```

The optimal solution was to produce 24 units of product A, 6 units of product B and use 4 units of additional resource R2. The total profit is 470.

## Branch and Bound Algorithm

Solving the mixed integer programming problem relies on the linear programming algorithm. It is very quick to solve the linear programming problem using the simplex algorithm. However, the solution to the linear programming problem is not necessarily an integer solution. The branch and bound algorithm is used to find the integer solution.

If we removed the integer constraints, we would get the following output from PuLP:

```
{'Product A': 23.333333, 'Product B': 6.6666667, 'Additional Resource R2': 3.3333333, 'Total Profit': 483.33332950000005}
```

As we can see from the output, `Continuous objective value is 483.333`, the first step is removing the integer constraints and solving the problem as a linear programming problem. This gives the upper bound on the solution.

Next step is turning the floating points into integers (rounding up and rounding down) and finding the best of them. This gives us a solution and we can see it in the log as `Integer solution of -470 found by DiveCoefficient`. 

The next step is to branch the problem or divide the problem into subproblems. Each subproblem works on a subspace of the original problem. We have a known solution and using linear programming and dropping the linear constraint can give us the upper bound for the subproblem. We can use the upper bound to prune the subproblem. If the upper bound is less than the known solution, we can prune the subproblem.

In the log, we can see that cuts were generated using the probing algorthm in the line `Probing was tried 1 times and created 2 cuts of which 0 were active after adding rounds of cuts`. However, the cuts were not helpful in tighening the solution we have so far and thus, the algorithm did not use them. Thus, our optimal solution was what was found using the `DiveCoefficient` algorithm.

## Plot of Bounds

If we visualize our constraints in 3d for the variables $x_A$, $x_B$ and $y$, we get the following plot:

![Plot of Bounds](/images/mip_plot.svg)

Green is our mixed integer solution and red is the solution where we drop the integer constraints. 

Source code for this article can be found [here](https://github.com/mochan-b/mixed-integer-programming).