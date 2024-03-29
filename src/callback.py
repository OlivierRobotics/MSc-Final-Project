# Modified from helper.py
# Copyright (c) 2018, The SenseAct Authors.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import builtins
import csv

def create_callback(shared_returns):
    builtins.shared_returns = shared_returns

    def kindred_callback(locals, globals):
        shared_returns = globals['__builtins__']['shared_returns']
        if locals['iters_so_far'] > 0:
            ep_rets = locals['seg']['ep_rets']
            ep_lens = locals['seg']['ep_lens']
            # create copy of policy object and save policy
            #pi = locals['pi']
            #pi.save('saved_policies/trpo01')
            if len(ep_rets):
                if not shared_returns is None:
                    shared_returns['write_lock'] = True
                    shared_returns['episodic_returns'] += ep_rets
                    shared_returns['episodic_lengths'] += ep_lens
                    shared_returns['write_lock'] = False
                    with open('experiment_data/no_reset_test_trpo03', 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        for data in zip(ep_rets, ep_lens):
                            csvwriter.writerow(data)
    return kindred_callback
