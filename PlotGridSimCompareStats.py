# ***** begin license block *****
# L2BM Simulator: Lazy Load-Balancing Multicast(L2BM) simulator
# copyright (c) 2016-2017, inria.
#
# this work was partially supported by the anr reflexion project (anr-14-ce28-0019).
# the authors alone are responsible for this work.
#
# see the file authors for details and contact information.
#
# this file is part of Lazy Load-Balancing Multicast(L2BM) simulator.
#
# this program is free software; you can redistribute it and/or modify it
# under the terms of the gnu general public license version 3 or later
# (the "gpl"), or the gnu lesser general public license version 3 or later
# (the "lgpl") as published by the free software foundation.
#
# you should have received a copy of the gnu general public license and
# gnu lesser general public license along with L2BM; see the file
# copying. if not, see <http://www.gnu.org/licenses/>.
#
# ***** end license block ***** */
# author: hardik soni <hardik.soni@inria.fr>

# !/usr/bin/python


import Utils as utils
import itertools
import time
import numpy as np
import sys
import SimulationStatsProcessor
import OtherStatsProcessor
import argparse
import scipy.stats as ss
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
import matplotlib.pyplot as plt
plt.style.use('classic')





# python ~/multicast-qos-python-may-2016/SimulationStatsProcessor.py
# --groups_parent_dirs /home/hsoni/qos-multicast-lb-compile/qos-multicast-lb,
#   /home/hsoni/qos-multicast-lb-compile/qos-multicast-sp
# --labels QoS-Multicast_LB,QoS-SSM
# --groups_start 5  --groups_stop 30  --groups_size_int 5
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups_parent_dirs", help="csv path dir containing group numbers' subdirs")
    parser.add_argument("--labels", help="experiments labels to for in order of groups_parent_dirs")
    parser.add_argument("--groups_start", help="starting number for multicast group size")
    parser.add_argument("--groups_stop", help="last number for multicast group size")
    parser.add_argument("--groups_size_int", help="interval for multicast group size")
    args = parser.parse_args()
    groups_parent_dirs_sim = ''
    groups_parent_dirs_grid = ''
    if args.groups_parent_dirs is not None and args.groups_start is not None \
            and args.groups_stop is not None and args.groups_size_int is not None \
            and args.labels is not None:
        groups_parent_dirs = args.groups_parent_dirs
        labels = args.labels
        start = args.groups_start
        stop = args.groups_stop
        inter = args.groups_size_int
        run_list = range(1, 21, 1)

    else:
        SimulationStatsProcessor.link_capacity = 100.0
        utils.working_dir = '/user/hsoni/home/'
        utils.bw_map_file = utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/ip-bw-qos-mapping-va.txt'
        groups_parent_dirs_sim = utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/dst,' \
                                 +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/dst-lb,' \
                                 +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/l2bm-10,' \
                                 +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/l2bm-40,' \
                                 +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/l2bm-60'

        groups_parent_dirs_grid = utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-75/dst,' \
                                  +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-75/lb,' \
                                  +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-10-10,' \
                                  +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-40-10,' \
                                  +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-60-10'
        labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.4,L2BM-0.6'
        start = 30
        stop = 70
        inter = 10
        run_list = range(1, 21, 1)
    plot_save_dir = '/user/hsoni/home/tnsm-review-sim/grid-sim-compare-plots/'
    utils.mkdir(plot_save_dir)
    color_dict = utils.get_color_dict(groups_parent_dirs_grid, labels)
    marker_dict = utils.get_marker_dict(groups_parent_dirs_grid, labels)
    SimulationStatsProcessor.plot_superimposed_datapoints_for_different_groups(groups_parent_dirs_sim,
                                                                               groups_parent_dirs_grid,
                                                                               plot_save_dir, labels, start, stop,
                                                                               inter, run_list, color_dict, marker_dict)
    OtherStatsProcessor.plot_superimposed_bw_accept_ratio_for_different_groups(groups_parent_dirs_sim,
                                                                               groups_parent_dirs_grid,
                                                                               plot_save_dir, labels, start, stop,
                                                                               inter, run_list, color_dict, marker_dict)
