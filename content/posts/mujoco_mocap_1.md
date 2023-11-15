+++
title = 'Mujoco Mocap Tutorial 1: Mocap object and keyboard movement'
author = 'Mochan Shrestha'
date = 2023-11-11T15:57:51-05:00
draft = false
+++

In this tutorial, we will go over the mocap object in mujoco and set up a simple xml file and python code where we control the mocap object using the keyboard.

All of the relevant code for this tutorial is found at the repo https://github.com/mochan-b/mujoco_mocap_tutorial

## MoCap in Mujoco

Mocap is a body in mujoco that does not have any physics associated with itself but can be used to manipulate other objects. As the name suggests, it can be used as a end point of a effector and the whole robot can be controlled so that the end effector is at the position of the mocap.

Obviously, mujoco itself will have to figure out the joint angles, velocities, forces etc to get the robot to the position that the end effector is in that position. Mujoco itself figures out those values using a method called inverse dynamics. 

## Mujoco XML file

Let's first start with the simple mujoco XML file that has the floor only. To view the XML file and interact with it we will be using mujoco python. Installing mujoco now is as simple as `pip install mujoco`.

The XML file just contains the floor and background.

```xml
<mujoco model="basic scene">
    <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                 markrgb="0.8 0.8 0.8" width="300" height="300"/>
        <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>

    <worldbody>
        <light pos="1 -1 1.5" dir="-1 1 -1" diffuse="0.5 0.5 0.5" directional="true"/>
        <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
    </worldbody>

</mujoco>
```

To run the XML file, you can use the basic python script

```python
import mujoco
import mujoco.viewer


def main():
    # Load the mujoco model basic.xml
    model = mujoco.MjModel.from_xml_path('basic.xml')
    data = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == '__main__':
    main()
```

![Mujoco Floor](/images/mujoco_floor.png)

## Adding a MoCap body

By adding the following to the `worldbody`, we have a mocap point. We use the site markeres to visualize the mocap axes.

```xml
<body mocap="true" name="mocap" pos="0.1 0.1 0.1">
    <site pos="0 0 0.075" size="0.003 0.003 0.1" type="box" name="mocap_left_site1" rgba="0 0 1 1"/>
    <site pos="0 0.075 0" size="0.003 0.1 0.003" type="box" name="mocap_left_site2" rgba="0 1 0 1"/>
    <site pos="0.075 0 0" size="0.1 0.003 0.003" type="box" name="mocap_left_site3" rgba="1 0 0 1"/>
</body>
```

![Mocap axes](/images/mujoco_mocap.png)

## Using the keyboard to move the point around

Finally, in this tutorial, we will use the keyboard to move the point around (in all 6 degrees of freedom). 

To be able to rotate an element in any axis, we need to be able to adjust the quaternion of the mocap. Since rotation in quaternions are just quaternion multiplications, the process is straightfoward using pyquaternion library.

```python
def rotate_quaternion(quat, axis, angle):
    """
    Rotate a quaternion by an angle around an axis
    """
    angle_rad = np.deg2rad(angle)
    axis = axis / np.linalg.norm(axis)
    q = pyq.Quaternion(quat)
    q = q * pyq.Quaternion(axis=axis, angle=angle_rad)
    return q.elements
```

Next is just adding the keyboard callback into the viewer and dealing with all of the 12 keys for adding and subtracting to the 6 degrees of fredom. 

We will use the arrow keys for up, down, left, right and numpad 0 and . for forward and back. For rotations, we will use the 6 keys that are ins, del, home, end, pg up and pg down for rotations along x, y and z respectively.

The code for this looks like the following:

```python
def key_callback(key):
    if key == 265:  # Up arrow
        data.mocap_pos[0, 2] += 0.01
    elif key == 264:  # Down arrow
        data.mocap_pos[0, 2] -= 0.01
    elif key == 263:  # Left arrow
        data.mocap_pos[0, 0] -= 0.01
    elif key == 262:  # Right arrow
        data.mocap_pos[0, 0] += 0.01
    elif key == 320:  # Numpad 0
        data.mocap_pos[0, 1] += 0.01
    elif key == 330:  # Numpad .
        data.mocap_pos[0, 1] -= 0.01
    elif key == 260:  # Insert
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [1, 0, 0], 10)
    elif key == 261:  # Home
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [1, 0, 0], -10)
    elif key == 268:  # Home
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 1, 0], 10)
    elif key == 269:  # End
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 1, 0], -10)
    elif key == 266:  # Page Up
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 0, 1], 10)
    elif key == 267:  # Page Down
        data.mocap_quat[0] = rotate_quaternion(data.mocap_quat[0], [0, 0, 1], -10)
    else:
        print(key)
```

And with this we can control the mocap using the keyboard in all 6 degrees of freedom and display it on the screen.

Right now it just moves around in space and doesn't affect anything. In the next tutorial, we will attach it to a robot and make the end effector of the robot move to our mocap position. 