# Adapted from ur_setups.py
# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.

"""Contains a setup for UR10 reacher environments. Specifies safety
box dimensions, joint limits to avoid self-collision etc."""

import numpy as np

setups = {
    'UR10_6dof':
              {
                  'host': '192.168.1.110',  # put UR5 Controller address here
                  #TODO: Update bounding box positions and angles based on UR10
                  'end_effector_low': np.array([-0.5, -0.7, 0.3]), # lower box bounding point - 100, 200, 300
                  'end_effector_high': np.array([-0.1, -0.2, 0.9]),  # upper box bounding point - 500, 700, 900
                  'angles_low':np.pi/180 * np.array( # used for joint angle bound checking
                      [ 30,
                       -180,
                       -160,
                       -90,
                        30,
                        -180
                       ]
                  ),
                  'angles_high':np.pi/180 * np.array(
                      [ 150,
                        0,
                        0,
                        50,
                        150,
                        180
                       ]
                  ),
                  'speed_max': 0.3,   # maximum joint speed magnitude using speedj
                  'accel_max': 1,      # maximum acceleration magnitude of the leading axis using speedj
                  'reset_speed_limit': 0.5,
                  'q_ref': np.pi/180 * np.array([ 70, -80, -120, 20, 100, 0]), # joint angles for start/reset pos
                  'box_bound_buffer': 0.001,
                  'angle_bound_buffer': 0.001,
                  'ik_params': # taken from https://www.universal-robots.com/how-tos-and-faqs/faq/ur-faq/parameters-for-calculations-of-kinematics-and-dynamics-45257/
                      (
                          0.1273,    # d1
                          -0.612,    # a2
                          -0.5723,   # a3
                          0.163941,  # d4
                          0.1157,    # d5
                          0.0922     # d6
                      )
              }
}