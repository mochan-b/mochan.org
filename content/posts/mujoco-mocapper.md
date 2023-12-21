+++
title = 'Mujoco Mocapper'
author = 'Mochan Shrestha'
date = 2023-12-21T03:21:00-05:00
draft = false
+++

Mujoco Mocapper is a tool that takes a mujoco file and adds mocap to it. It can be installed via the `pip` command.

In the previous articles [here](https://mochan.org/posts/mujoco_mocap_1/) and [here](https://mochan.org/posts/mujoco-mocap-2/), we discussed how to add mocap to a mujoco XML fie. The steps are very manual and tediuous and can be error prone and time consuming. Using the `mujoco-mocapper` tool, we can automate the process and make it easier to add mocap to a mujoco XML file.

The following links for the tool:
1. PyPi: [https://pypi.org/project/mujoco-mocapper/](https://pypi.org/project/mujoco-mocapper/)
2. Github: [https://github.com/mochan-b/mujoco_mocapper](https://github.com/mochan-b/mujoco_mocapper)

![](/images/mujoco-mocapper-icon.png)

## Installation and Usage

The tool can be installed via the `pip` command.

```bash
pip install mujoco_mocapper
```

and then you can run the tool via the `mujoco-mocapper` command.

```bash
mujoco-mocapper <input-file> <body> <output-file>
```

## Tool Steps

The tool does the following steps:
1. Runs mujoco and finds the global position of the body we want to mocap
2. Opens the input XML file and then adds the mocap related elements to the XML file and saves it as an XML file
3. Runs mujoco on the new XML file with the keyboard controlled mocap

## Examples

Here are some examples of the tool and mocapping some examples from the [mujoco-menagerie](https://github.com/google-deepmind/mujoco_menagerie) repo.

### Example 1: UR10e

```bash
mujoco-mocapper mujoco_menagerie/universal_robots_ur10e/scene.xml wrist_3_link mujoco_menagerie/universal_robots_ur10e/scene_mc.xml
```
![UR10e](/images/ur10e_mocap.png)

### Example 2: Skydio

```bash
mujoco-mocapper mujoco_menagerie/skydio_x2/scene.xml x2 mujoco_menagerie/skydio_x2/scene_mc.xml
```

![Skydio](/images/skydio_mocap.png)

## Limitations and Future Features

The tool is still in its early stages.

1. Keyframes are not part of the initalization but getting the mocap positions for the keyframes would be very useful.
1. Support for multiple mocaps/bodies etc.
1. The tool only supports keyboard controlled mocap. It would be nice to support a more general mocap and ability to save the current mocap and joint positions and replay them back.
1. Deal with the assets file locations that require more advanced XML parsing.