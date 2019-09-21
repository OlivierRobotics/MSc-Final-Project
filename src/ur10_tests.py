# Adapted from ur5_reacher_6d.py
# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.

import time
import copy
import numpy as np
import sys

import baselines.common.tf_util as U
from multiprocessing import Process, Value, Manager
from baselines.trpo_mpi.trpo_mpi import learn
from baselines.ppo1.mlp_policy import MlpPolicy

from senseact.envs.ur.reacher_env import ReacherEnv
from senseact.utils import tf_set_seeds, NormalizedEnv

from tensorflow.train import Saver
from tensorflow.saved_model import simple_save, loader

#sys.path.append("/home/oli/SenseAct/examples/advanced")
#from helper import create_callback
from callback import create_callback
from run_policy import run_policy

from senseact.devices.ur import ur_utils
from senseact import utils

import builtins
import csv

# an environment that allows points to be selected on a x_points x y_points x z _points grid within the end effector bounds and tests each point num_test times
class GridTestEnv(ReacherEnv):
    def __init__(self,
                 setup,
                 host=None,
                 dof=6,
                 control_type='position',
                 derivative_type='none',
                 reset_type='random',
                 reward_type='linear',
                 deriv_action_max=10,
                 first_deriv_max=10,  # used only with second derivative control
                 vel_penalty=0,
                 obs_history=1,
                 actuation_sync_period=1,
                 episode_length_time=None,
                 episode_length_step=None,
                 rllab_box = False,
                 servoj_t=ur_utils.COMMANDS['SERVOJ']['default']['t'],
                 servoj_gain=ur_utils.COMMANDS['SERVOJ']['default']['gain'],
                 speedj_a=ur_utils.COMMANDS['SPEEDJ']['default']['a'],
                 speedj_t_min=ur_utils.COMMANDS['SPEEDJ']['default']['t_min'],
                 movej_t=2, # used for resetting
                 accel_max=None,
                 speed_max=None,
                 dt=0.008,
                 delay=0.0,  # to simulate extra delay in the system
                 x_points=10,
                 y_points=10,
                 z_points=10,
                 num_test=10,
                 **kwargs):
        
        assert(x_points > 0)
        assert(y_points > 0)
        assert(num_test > 0)

        self._x_points = x_points
        self._y_points = y_points
        self._z_points = z_points
        self._num_test = num_test

        self._target_generator_ = self._target_generator_()

        super(GridTestEnv, self).__init__(setup=setup,
                                         host=host,
                                         dof=dof,
                                         control_type=control_type,
                                         derivative_type=derivative_type,
                                         target_type='position',
                                         reset_type=reset_type,
                                         reward_type=reward_type,
                                         deriv_action_max=deriv_action_max,
                                         first_deriv_max=first_deriv_max,  # used only with second derivative control
                                         vel_penalty=vel_penalty,
                                         obs_history=obs_history,
                                         actuation_sync_period=actuation_sync_period,
                                         episode_length_time=episode_length_time,
                                         episode_length_step=episode_length_step,
                                         rllab_box = rllab_box,
                                         servoj_t=servoj_t,
                                         servoj_gain=servoj_gain,
                                         speedj_a=speedj_a,
                                         speedj_t_min=speedj_t_min,
                                         movej_t=movej_t, # used for resetting
                                         accel_max=accel_max,
                                         speed_max=speed_max,
                                         dt=dt,
                                         delay=delay,  # to simulate extra delay in the system
                                         **kwargs)

    def _reset_(self):
        """Resets the environment episode.

        Moves the arm to either fixed reference or random position and
        generates a new target from _target_generator_.
        """
        print("Resetting")

        x_target = self._target_generator_.__next__()
        np.copyto(self._x_target_, x_target)
        self._target_ = self._x_target_[self._end_effector_indices]

        self._action_ = self._rand_obj_.uniform(self._action_low, self._action_high)
        self._cmd_prev_ = np.zeros(len(self._action_low))  # to be used with derivative control of velocity
        if self._reset_type != 'none':
            if self._reset_type == 'random':
                reset_angles, _ = self._pick_random_angles_()
            elif self._reset_type == 'zero':
                reset_angles = self._q_ref[self._joint_indices]
            self._reset_arm(reset_angles)

        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(
            self._rand_obj_.get_state()
        )
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))

        print("Reset done")

    def _target_generator_(self):
        # increments for each dimension
        x_inc = (self._end_effector_high[0] - self._end_effector_low[0]) / (self._x_points+1)
        y_inc = (self._end_effector_high[1] - self._end_effector_low[1]) / (self._y_points+1)
        z_inc = (self._end_effector_high[2] - self._end_effector_low[2]) / (self._z_points+1)

        # lists of x, y, z coords
        x_points = [self._end_effector_low[0] + x_inc * x_point for x_point in range(1, self._x_points+2)]
        y_points = [self._end_effector_low[1] + y_inc * y_point for y_point in range(1, self._y_points+2)]
        z_points = [self._end_effector_low[2] + z_inc * z_point for z_point in range(1, self._z_points+2)]

        for x in range(self._x_points):
            for y in range(self._y_points):
                for z in range(self._z_points):
                    for test in range(self._num_test):
                        yield x_points[x], y_points[y], z_points[z]

# callback to use for logging 
def grid_test_callback(locals, globals):
    shared_returns = globals['__builtins__']['shared_returns']
    if locals['iters_so_far'] > 0:
        ep_rets = locals['seg']['ep_rets']
        ep_lens = locals['seg']['ep_lens']
        target = locals['env']._x_target_

        if len(ep_rets):
            if not shared_returns is None:
                shared_returns['write_lock'] = True
                shared_returns['episodic_returns'] += ep_rets
                shared_returns['episodic_lengths'] += ep_lens
                shared_returns['write_lock'] = False
                with open('experiment_data/gridtest.csv', 'a', newline='') as csvfile:
                    csvwriter = csv.writer(csvfile)
                    row = [np.mean(ep_rets), *target]
                    csvwriter.writerow(row)

# Load a policy from policy_path and runs a grid test on it with x, y, z_points points testing each point num_test times
def run_grid_test(x_points, y_points, z_points, num_test, policy_path):
    # use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))

    # set up coordination between eps per iteration and num_test
    episode_length_time = 4.0
    dt = 0.04
    timesteps_per_ep = episode_length_time / dt
    timesteps_per_batch = int(timesteps_per_ep * num_test) # small extra time just in case


    # Create GridTest environment
    env = GridTestEnv(
            setup="UR10_6dof",
            host=None,
            dof=6,
            control_type="velocity",
            reset_type="zero",
            reward_type="precision",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4, # was 1.4
            speed_max=0.3, # was 0.3
            speedj_a=1.4,
            episode_length_time=episode_length_time,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=dt,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=rand_state,
            x_points=x_points,
            y_points=y_points,
            z_points=z_points,
            num_test=num_test
        )
    env = NormalizedEnv(env)
    # Start environment processes
    env.start()
    # Create baselines TRPO policy function
    sess = U.single_threaded_session()
    sess.__enter__()

    # Create and start plotting process
    plot_running = Value('i', 1)
    shared_returns = Manager().dict({"write_lock": False,
                                     "episodic_returns": [],
                                     "episodic_lengths": [], })
    builtins.shared_returns = shared_returns

    # Spawn plotting process
    pp = Process(target=plot_ur5_reacher, args=(env, 2048, shared_returns, plot_running))
    pp.start()

    # Run TRPO policy
    run_policy(network='mlp', 
          num_layers=2, # these are network_kwargs for the MLP network
          num_hidden=64,
          env=env, 
          total_timesteps=50000, #Originally 200,000
          timesteps_per_batch=timesteps_per_batch,
          callback=grid_test_callback,
          load_path=policy_path
          )

    # Safely terminate plotter process
    plot_running.value = 0  # shutdown plotting process
    time.sleep(2)
    pp.join()

    env.close()

def main():
    # use fixed random state
    rand_state = np.random.RandomState(1).get_state()
    np.random.set_state(rand_state)
    tf_set_seeds(np.random.randint(1, 2**31 - 1))

    # Create UR5 Reacher2D environment
    env = ReacherEnv(
            setup="UR10_6dof",
            host=None,
            dof=6,
            control_type="velocity",
            target_type="position",
            reset_type="zero",
            reward_type="precision",
            derivative_type="none",
            deriv_action_max=5,
            first_deriv_max=2,
            accel_max=1.4, # was 1.4
            speed_max=0.3, # was 0.3
            speedj_a=1.4,
            episode_length_time=4.0,
            episode_length_step=None,
            actuation_sync_period=1,
            dt=0.04,
            run_mode="multiprocess",
            rllab_box=False,
            movej_t=2.0,
            delay=0.0,
            random_state=rand_state
        )
    env = NormalizedEnv(env)
    # Start environment processes
    env.start()
    # Create baselines TRPO policy function
    sess = U.single_threaded_session()
    sess.__enter__()

    # Load previously trained model if it exists


    # No longer needed
    """def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)"""

    # Create and start plotting process
    plot_running = Value('i', 1)
    shared_returns = Manager().dict({"write_lock": False,
                                     "episodic_returns": [],
                                     "episodic_lengths": [], })
    # Spawn plotting process
    pp = Process(target=plot_ur5_reacher, args=(env, 2048, shared_returns, plot_running))
    pp.start()

    # Create callback function for logging data from baselines TRPO learn
    kindred_callback = create_callback(shared_returns)

    # Train baselines TRPO
    learn(network='mlp', 
    	  num_layers=2, # these are network_kwargs for the MLP network
    	  num_hidden=64,
    	  env=env, 
    	  total_timesteps=50000, #Originally 200,000
          timesteps_per_batch=1000,
          max_kl=0.05,
          cg_iters=10,
          cg_damping=0.1,
          vf_iters=5,
          vf_stepsize=0.001,
          gamma=0.995,
          lam=0.995,
          callback=kindred_callback,
          load_path=None,
          save_path='saved_policies/trpo02',
          )

    # Safely terminate plotter process
    plot_running.value = 0  # shutdown ploting process
    time.sleep(2)
    pp.join()

    env.close()

class MovingPointEnv(ReacherEnv):
    def __init__(self,
                 setup,
                 host=None,
                 dof=6,
                 control_type='position',
                 derivative_type='none',
                 reset_type='random',
                 reward_type='linear',
                 deriv_action_max=10,
                 first_deriv_max=10,  # used only with second derivative control
                 vel_penalty=0,
                 obs_history=1,
                 actuation_sync_period=1,
                 episode_length_time=None,
                 episode_length_step=None,
                 rllab_box = False,
                 servoj_t=ur_utils.COMMANDS['SERVOJ']['default']['t'],
                 servoj_gain=ur_utils.COMMANDS['SERVOJ']['default']['gain'],
                 speedj_a=ur_utils.COMMANDS['SPEEDJ']['default']['a'],
                 speedj_t_min=ur_utils.COMMANDS['SPEEDJ']['default']['t_min'],
                 movej_t=2, # used for resetting
                 accel_max=None,
                 speed_max=None,
                 dt=0.008,
                 delay=0.0,  # to simulate extra delay in the system
                 move_shape='circle', # circle or line
                 move_vel=0.1, # velocity of moving point in m/s
                 line_midpoint=[0, 0, 0],
                 line_length=0.5,
                 circle_radius=0.3,
                 **kwargs):
        
        assert(move_shape == 'circle' or move_shape == 'line')
        assert(line_midpoint.length() == 3)
        assert(line_length > 0)
        assert(circle_radius > 0)
        self._move_shape = move_shape
        self._move_vel = move_vel
        self._line_midpoint_ = line_midpoint
        self._line_length_ = line_length
        self._circle_radius_ = circle_radius

        if(move_shape == 'circle'):
            self._move_generator_ = self._circle_generator_()
        else if(move_shape == 'line'):
            self._move_generator_ = self._line_generator()

        super(MovingPointEnv, self).__init__(setup=setup,
                                         host=host,
                                         dof=dof,
                                         control_type=control_type,
                                         derivative_type=derivative_type,
                                         target_type='position',
                                         reset_type=reset_type,
                                         reward_type=reward_type,
                                         deriv_action_max=deriv_action_max,
                                         first_deriv_max=first_deriv_max,  # used only with second derivative control
                                         vel_penalty=vel_penalty,
                                         obs_history=obs_history,
                                         actuation_sync_period=actuation_sync_period,
                                         episode_length_time=episode_length_time,
                                         episode_length_step=episode_length_step,
                                         rllab_box = rllab_box,
                                         servoj_t=servoj_t,
                                         servoj_gain=servoj_gain,
                                         speedj_a=speedj_a,
                                         speedj_t_min=speedj_t_min,
                                         movej_t=movej_t, # used for resetting
                                         accel_max=accel_max,
                                         speed_max=speed_max,
                                         dt=dt,
                                         delay=delay,  # to simulate extra delay in the system
                                         **kwargs)


    # overrides step function in RTRLBaseEnv to allow for update of target each step
    def step(self, action):
        """Optional step function for OpenAI Gym compatibility.

        Returns: a tuple (observation, reward,  {} ('info', for gym compatibility))
        """
        # Set the desired action
        self.act(action)
        # Update target
        self._x_target_ = _move_generator_.__next__()
        # Wait for one time-step
        next_obs, reward, done = self.sense_wait()
        return next_obs, reward, done, {}

    def _line_generator_(self):
        

    def _reset_(self):
        """Resets the environment episode.

        Moves the arm to either fixed reference or random position and
        generates a new target from _target_generator_.
        """
        print("Resetting")
        if(self._move_shape_ == 'circle'):
            x_target = self._end_effector_high - self._end_effector_low + np.array([self._circle_radius_, 0, 0])
        else if(self._move_shape_ == 'line'):
            x_target = self._end_effector_high - self._end_effector_low + np.array(self._line_midpoint_)
        np.copyto(self._x_target_, x_target)
        self._target_ = self._x_target_[self._end_effector_indices]

        self._action_ = self._rand_obj_.uniform(self._action_low, self._action_high)
        self._cmd_prev_ = np.zeros(len(self._action_low))  # to be used with derivative control of velocity
        if self._reset_type != 'none':
            if self._reset_type == 'random':
                reset_angles, _ = self._pick_random_angles_()
            elif self._reset_type == 'zero':
                reset_angles = self._q_ref[self._joint_indices]
            self._reset_arm(reset_angles)

        rand_state_array_type, rand_state_array_size, rand_state_array = utils.get_random_state_array(
            self._rand_obj_.get_state()
        )
        np.copyto(self._shared_rstate_array_, np.frombuffer(rand_state_array, dtype=rand_state_array_type))

        print("Reset done")


def plot_ur5_reacher(env, batch_size, shared_returns, plot_running):
    """Helper process for visualize the tasks and episodic returns.

    Args:
        env: An instance of ReacherEnv
        batch_size: An int representing timesteps_per_batch provided to the PPO learn function
        shared_returns: A manager dictionary object containing `episodic returns` and `episodic lengths`
        plot_running: A multiprocessing Value object containing 0/1.
            1: Continue plotting, 0: Terminate plotting loop
    """
    print ("Started plotting routine")
    import matplotlib.pyplot as plt
    plt.ion()
    time.sleep(5.0)
    fig = plt.figure(figsize=(20, 6))
    ax1 = fig.add_subplot(131)
    hl1, = ax1.plot([], [], markersize=10, marker="o", color='r')
    hl2, = ax1.plot([], [], markersize=10, marker="o", color='b')
    ax1.set_xlabel("X", fontsize=14)
    h = ax1.set_ylabel("Y", fontsize=14)
    h.set_rotation(0)
    ax3 = fig.add_subplot(132)
    hl3, = ax3.plot([], [], markersize=10, marker="o", color='r')
    hl4, = ax3.plot([], [], markersize=10, marker="o", color='b')
    ax3.set_xlabel("Z", fontsize=14)
    h = ax3.set_ylabel("Y", fontsize=14)
    h.set_rotation(0)
    ax2 = fig.add_subplot(133)
    hl11, = ax2.plot([], [])
    count = 0
    old_size = len(shared_returns['episodic_returns'])
    while plot_running.value:
        plt.suptitle("Reward: {:.2f}".format(env._reward_.value), x=0.375, fontsize=14)
        hl1.set_ydata([env._x_target_[1]])
        hl1.set_xdata([env._x_target_[2]])
        hl2.set_ydata([env._x_[1]])
        hl2.set_xdata([env._x_[2]])
        ax1.set_ylim([env._end_effector_low[1], env._end_effector_high[1]])
        ax1.set_xlim([env._end_effector_low[2], env._end_effector_high[2]])
        ax1.set_title("X-Y plane", fontsize=14)
        ax1.set_xlim(ax1.get_xlim()[::-1])
        ax1.set_ylim(ax1.get_ylim()[::-1])

        hl3.set_ydata([env._x_target_[1]])
        hl3.set_xdata([env._x_target_[0]])
        hl4.set_ydata([env._x_[1]])
        hl4.set_xdata([env._x_[0]])
        ax3.set_ylim([env._end_effector_low[1], env._end_effector_high[1]])
        ax3.set_xlim([env._end_effector_low[0], env._end_effector_high[0]])
        ax3.set_title("Y-Z plane", fontsize=14)
        ax3.set_xlim(ax3.get_xlim()[::-1])
        ax3.set_ylim(ax3.get_ylim()[::-1])

        # make a copy of the whole dict to avoid episode_returns and episodic_lengths getting desync
        # while plotting
        copied_returns = copy.deepcopy(shared_returns)
        if not copied_returns['write_lock'] and  len(copied_returns['episodic_returns']) > old_size:
            # plot learning curve
            returns = np.array(copied_returns['episodic_returns'])
            old_size = len(copied_returns['episodic_returns'])
            window_size_steps = 5000
            x_tick = 1000

            if copied_returns['episodic_lengths']:
                ep_lens = np.array(copied_returns['episodic_lengths'])
            else:
                ep_lens = batch_size * np.arange(len(returns))
            cum_episode_lengths = np.cumsum(ep_lens)

            if cum_episode_lengths[-1] >= x_tick:
                steps_show = np.arange(x_tick, cum_episode_lengths[-1] + 1, x_tick)
                rets = []

                for i in range(len(steps_show)):
                    rets_in_window = returns[(cum_episode_lengths > max(0, x_tick * (i + 1) - window_size_steps)) *
                                             (cum_episode_lengths < x_tick * (i + 1))]
                    if rets_in_window.any():
                        rets.append(np.mean(rets_in_window))

                hl11.set_xdata(np.arange(1, len(rets) + 1) * x_tick)
                ax2.set_xlim([x_tick, len(rets) * x_tick])
                hl11.set_ydata(rets)
                ax2.set_ylim([np.min(rets), np.max(rets) + 50])
        time.sleep(0.01)
        fig.canvas.draw()
        fig.canvas.flush_events()
        count += 1


if __name__ == '__main__':
    run_grid_test(10, 10, 10, 3, 'saved_policies/trpo01/trpo01')