# adapted from baselines.trpo_mpi.trpo_mpi

from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.input import observation_placeholder
from baselines.common.policies import build_policy
from contextlib import contextmanager
import os.path
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

def traj_segment_generator(pi, env, horizon, stochastic):
    # Initialize state variables
    print("Horizon = %i"%horizon)
    t = 0
    ac = env.action_space.sample()
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred, _, _ = pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def run_policy(*,
        network,
        env,
        total_timesteps,
        timesteps_per_batch=1024, # what to train on
        seed=None,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        **network_kwargs
        ):
    '''
    run a policy function

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))


    policy = build_policy(env, network, value_network='copy', **network_kwargs)
    set_global_seeds(seed)

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space

    ob = observation_placeholder(ob_space)
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)
    with tf.variable_scope("oldpi"):
        oldpi = policy(observ_placeholder=ob)

    ac = pi.pdtype.sample_placeholder([None])

    all_var_list = get_trainable_variables("pi")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    var_list = get_pi_trainable_variables("pi")

    get_flat = U.GetFlat(var_list)
    set_from_flat = U.SetFromFlat(var_list)

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    U.initialize()
    if load_path is not None and os.path.exists(load_path+'.meta'):
        pi.load(load_path)
        print('Loaded from '+load_path)

    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)

    set_from_flat(th_init)
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # nothing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    while True:
        target = env._x_target_
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        if rank==0:
            logger.dump_tabular()

    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]

