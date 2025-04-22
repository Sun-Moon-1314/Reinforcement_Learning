# -*- coding: utf-8 -*-
"""
@File    : mujoco_test.py
@Time    : 2025/4/20 11:11
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
import mujoco.viewer
import mujoco.viewer as mujoco_viewer
import numpy as np
import time

import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path('/Users/zhangjian/PycharmProjects/Reinforcement_Learning/.venv_test/lib/python3.9/site-packages/mujoco/testdata/model.xml')
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
  # Close the viewer automatically after 30 wall-seconds.
  start = time.time()
  while viewer.is_running() and time.time() - start < 30:
    step_start = time.time()

    # mj_step can be replaced with code that also evaluates
    # a policy and applies a control signal before stepping the physics.
    mujoco.mj_step(m, d)

    # Example modification of a viewer option: toggle contact points every two seconds.
    with viewer.lock():
      viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(d.time % 2)

    # Pick up changes to the physics state, apply perturbations, update options from GUI.
    viewer.sync()

    # Rudimentary time keeping, will drift relative to wall clock.
    time_until_next_step = m.opt.timestep - (time.time() - step_start)
    if time_until_next_step > 0:
      time.sleep(time_until_next_step)