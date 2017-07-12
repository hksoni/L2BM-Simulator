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



from SimulationStatsProcessor import *
from OtherStatsProcessor import *
import argparse

import matplotlib
matplotlib.use('Agg')
from Utils import *
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
import matplotlib.pyplot as plt

plt.style.use('classic')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--working_dir", help="path dir containing group numbers' subdirs")
    parser.add_argument("--groups_dirs", help="subdirs in working dir or relative to it")
    parser.add_argument("--labels", help="experiments labels to for in order of groups_parent_dirs")
    parser.add_argument("--groups_start", help="starting number for multicast group size")
    parser.add_argument("--groups_stop", help="last number for multicast group size")
    parser.add_argument("--groups_size_int", help="interval for multicast group size")
    args = parser.parse_args()
    wo_churn_congested_sim_dirs = ''
    if args.groups_dirs is not None and args.groups_start is not None \
            and args.groups_stop is not None and args.groups_size_int is not None \
            and args.labels is not None:

        for sub in args.groups_dirs:
            wo_churn_congested_sim_dirs += args.working_dir+"/"+sub+","
        wo_churn_congested_sim_dirs = wo_churn_congested_sim_dirs[:-1]
        labels = args.labels
        start = args.groups_start
        stop = args.groups_stop
        inter = args.groups_size_int
        run_list = range(1, 21, 1)
    else:
        utils.working_dir = '/user/hsoni/home/tnsm-review-sim/1G/'
        # wo_churn_congested_sim_dirs = utils.working_dir+'/wo-churn-congested/dst,' \
        #                    +utils.working_dir+'/wo-churn-congested/dst-lb,' \
        #                    +utils.working_dir+'/wo-churn-congested/l2bm-10,' \
        #                    +utils.working_dir+'/wo-churn-congested/l2bm-20,' \
        #                    +utils.working_dir+'/wo-churn-congested/l2bm-30,' \
        #                    +utils.working_dir+'/wo-churn-congested/l2bm-40,' \
        #                    +utils.working_dir+'/wo-churn-congested/l2bm-50,' \
        #                    +utils.working_dir+'/wo-churn-congested/l2bm-60'
        wo_churn_congested_sim_dirs = utils.working_dir+'/wo-churn-congested/dst,' \
                                      +utils.working_dir+'/wo-churn-congested/dst-lb,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-10,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-40,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-60,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-20,' \
        # labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.2,L2BM-0.3,L2BM-0.4,L2BM-0.5,L2BM-0.6'
        labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.4,L2BM-0.6,L2BM-0.2'
        start = 10
        stop = 150
        inter = 10
        link_capacity = 1000
        # run_list = range(1, 21, 1)
        run_list = range(500)
    osp_wo_c = OtherStatsProcessorWOChurn()
    plot_save_dir = utils.working_dir+'/wo-churn-congested/plots/'
    mkdir(plot_save_dir)
    color_dict = utils.get_color_dict(wo_churn_congested_sim_dirs, labels)
    marker_dict = utils.get_marker_dict(wo_churn_congested_sim_dirs, labels)
    color = plot_link_util_metrics_for_different_groups('sim', wo_churn_congested_sim_dirs, plot_save_dir, labels,
                                                        start, stop, inter, run_list, color_dict, marker_dict)
    osp_wo_c.plot_other_metrics_for_different_groups(wo_churn_congested_sim_dirs, plot_save_dir, labels, start, stop,
                                                     inter, run_list, color_dict, marker_dict)