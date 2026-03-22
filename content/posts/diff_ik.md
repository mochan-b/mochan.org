+++
title = 'Differential IK and Mujoco Mocap'
date = 2026-03-22T14:45:02-04:00
draft = false
+++

Differential inverse kinematics, or differential IK, controls a robot by solving for small joint motions that produce a desired instantaneous motion of the end effector rather than solving for a full joint configuration all at once. In practice, this means the controller takes a commanded Cartesian velocity, such as "move the tool slightly forward" or "rotate the wrist a little," and uses the robot's Jacobian to compute the joint velocities that best realize that motion at the current pose. This makes differential IK a natural fit for teleoperation and interactive control, where the goal is smooth local motion instead of jumping directly to a single target pose.

The [teleop using gamepad](https://mochan.dev/robot_gamepad/) web application uses differential IK to control the robot. In the project, each control loop reads the current gamepad state and converts it into a 6D end-effector velocity command of the form $[\dot{x}, \dot{y}, \dot{z}, \dot{\phi}, \dot{\theta}, \dot{\psi}]$. The application then evaluates the Jacobian at the current end-effector pose, solves the damped least-squares update for the joint velocities, and integrates that result into the arm's position-control targets before stepping the MuJoCo simulation forward. In other words, the joystick does not command a final pose directly; it continuously requests small Cartesian motions, and the solver updates the joints a little bit at a time to realize that motion smoothly.

Mathematically, differential IK starts from the local kinematic relationship $\dot{x} = J(q)\dot{q} $, where $ \dot{x} $ is the desired end-effector twist, $ \dot{q} $ is the vector of joint velocities, and $ J(q) $ is the Jacobian evaluated at the current joint configuration $ q $. Because the Jacobian is often non-square or poorly conditioned, we usually solve for $ \dot{q} $ with a damped least-squares pseudoinverse:

$$
\dot{q} = J^T (J J^T + \lambda^2 I)^{-1} \dot{x}
$$

This finds a joint motion that best matches the commanded Cartesian motion while remaining numerically stable near singularities. Once $ \dot{q} $ is computed, the controller advances the robot by a small timestep using

$$
q_{t+\Delta t} \approx q_t + \dot{q}\Delta t
$$

and then repeats this process at the next control cycle.

## Why Not Solve for Position Directly?

An alternative design would be to read the current end-effector pose, convert the input into a slightly shifted target pose, solve a standard IK problem for that new pose, and then command the arm to those joint angles. That approach is valid, and in fact it is closely related to differential IK when the pose changes are very small. The difference is that full pose IK tries to solve a new configuration-level problem at every control step, whereas differential IK works directly with the instantaneous motion that the joystick is asking for.

For teleoperation using a controller, differential IK is usually the better fit because a joystick naturally expresses velocity rather than destination. When the stick is centered, the commanded end-effector velocity is zero; when the stick is pushed farther, the commanded motion becomes faster. Solving in velocity space also tends to produce smoother motion for redundant arms, since the controller makes small local updates instead of repeatedly searching for a fresh joint configuration that satisfies the same pose. In practice, this reduces abrupt jumps between IK solutions, is computationally cheaper than repeatedly solving a full nonlinear pose IK problem, and behaves more gracefully near singularities or temporarily unreachable commands because the damped least-squares solve regularizes the motion instead of trying to force an exact pose match on every frame.

## MuJoCo Mocap Teleoperation

MuJoCo supports a different style of teleoperation through mocap bodies and welding. This is the technique used in this [tool](https://mochan.org/posts/mujoco-mocapper/) and tutorials [1](https://mochan.org/posts/mujoco_mocap_1/) and [2](https://mochan.org/posts/mujoco-mocap-2/). A mocap body is a special body attached directly to the world with no joints. By itself, a mocap body is not a dynamic rigid body that the simulator integrates forward in time; it is an externally specified kinematic target. The well-behaved approach is to connect a regular dynamic body to the mocap body with a `weld` equality constraint and then move the mocap target, letting the physics solver pull the dynamic system along. 

Under the hood, MuJoCo treats the gap between the mocap target and the welded body as a soft 6D constraint and resolves it through the dynamics solver, the part of the simulator that computes how forces and constraints produce accelerations and motion over the next timestep. Mathematically, the weld contributes a 6-dimensional residual $r(q)$ made of three translational errors and three rotational errors. Instead of measuring rotation mismatch with roll, pitch, and yaw differences, MuJoCo encodes it as a 3D vector pointing along the axis of rotation, with magnitude based on the angle difference through $\sin(\theta/2)$. It then computes a Jacobian $J(q)$ that describes how small changes in the robot's joints would change that weld error, that is, $J(q) = \partial r / \partial q$. A useful mental model is a 6D virtual spring-damper between the mocap target and the welded body, but MuJoCo implements this through its soft-constraint solver rather than by explicitly integrating a standalone spring. In the modeling chapter, MuJoCo describes the reference acceleration as

$$
a_{\mathrm{ref}} = -b v - k r,
$$

Here, $b$ and $k$ are the internal damping and stiffness terms of the constraint's reference acceleration, while $v$ is the constraint velocity and $r$ is the constraint violation. In MuJoCo, you usually do not set $b$ and $k$ directly. Instead, `solref` is the user-facing parameter that defines the reference dynamics, usually in terms of a time constant and damping ratio or, in the direct format, stiffness and damping. The `solimp` parameter defines the constraint impedance, meaning how strongly the constraint is allowed to act as the violation changes. MuJoCo then derives the effective $b$ and $k$ used by the solver from `solref`, `solimp`, and the current constraint state. The resulting weld constraint force becomes part of MuJoCo's general constrained equation of motion by contributing to the joint-space constraint term $J(q)^T f$:

$$
M(q)\dot{v} + c(q, v) = \tau + J(q)^T f.
$$

This equation is MuJoCo's general forward dynamics equation for constrained systems in generalized coordinates: it balances inertia and bias forces on the left against applied and constraint forces on the right, then solves for the resulting acceleration $\dot{v}$. Mocap teleoperation does not introduce a new equation; rather, the weld contributes some of the overall constraint force represented by the term $J(q)^T f$ on the right-hand side. Other constraints, such as contacts, joint limits, and other equality constraints, can also contribute to that same term.

| Symbol | Meaning |
| --- | --- |
| $q$ | The robot joint positions or generalized coordinates. |
| $v$ | The generalized velocity, which for a simple robot arm is usually the joint velocity vector. |
| $\dot{v}$ | The generalized acceleration, or joint acceleration. |
| $M(q)$ | The mass or inertia matrix, which describes how hard it is to accelerate the robot in different joint-space directions. |
| $c(q, v)$ | The bias forces from effects such as gravity, Coriolis forces, and centrifugal forces. |
| $\tau$ | The applied generalized force, including actuator, passive, and user-applied forces. |
| $J(q)$ | The constraint Jacobian, which maps joint-space motion into motion of the constraint error. |
| $J(q)^T f$ | The equivalent joint-space force produced by all active constraints acting through the Jacobian transpose. |
| $f$ | The vector of constraint-space forces for the currently active constraints; in the mocap teleoperation case, the weld contributes part of this vector. |

One way to summarize the MuJoCo mocap teleoperation loop is as the following algorithm:

1. Set the mocap body's target position and orientation for the current control step.
2. Compute the weld constraint error $r(q)$ between the mocap body and the welded robot body.
3. Compute the constraint Jacobian $J(q)$, which describes how joint motion would change that error.
4. Use `solref` and `solimp` to determine the effective reference dynamics of the soft constraint, producing a reference acceleration of the form $a_{\mathrm{ref}} = -b v - k r$.
5. Solve for the constraint force vector $f$ together with the rest of the simulator's contacts and constraints.
6. Extract the weld's contribution as part of the overall joint-space constraint force $J(q)^T f$.
7. Combine actuator torques, constraint forces, gravity, and other dynamic effects in the forward dynamics equation to compute $\dot{v}$.
8. Integrate $\dot{v}$ and $v$ over the timestep to update the robot's joint velocities and positions.
9. Repeat at the next simulation step with the next mocap target.

This is the key difference from differential IK. Differential IK starts with a desired end-effector twist and explicitly computes a joint velocity update that best realizes that motion locally. MuJoCo mocap teleoperation starts with a target pose in Cartesian space and lets the dynamics engine determine what motion actually happens under inertia, contacts, joint limits, actuator behavior, and constraint softness. As a result, a heavy arm may lag behind the target, collisions can overpower the weld, and unreachable targets show up as competing physical forces instead of a failing kinematic solve. Differential IK is a direct kinematic controller; mocap teleoperation is a target-pose-plus-forward-dynamics controller.

### References

- [MuJoCo XML reference: mocap bodies](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [MuJoCo computation chapter: equality constraints and weld residuals](https://mujoco.readthedocs.io/en/stable/computation/)
- [MuJoCo modeling chapter: mocap bodies, weld constraints, solref, and solimp](https://mujoco.readthedocs.io/en/stable/modeling.html)
