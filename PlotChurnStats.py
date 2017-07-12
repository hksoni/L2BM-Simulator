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
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
import matplotlib.pyplot as plt
plt.style.use('classic')
from OtherStatsProcessor import *
from ChurnStatsProcessor import *



# python ~/multicast-qos-python-may-2016/SimulationStatsProcessor.py
# --groups_parent_dirs /home/hsoni/qos-multicast-lb-compile/qos-multicast-lb,
#   /home/hsoni/qos-multicast-lb-compile/qos-multicast-sp
# --labels QoS-Multicast_LB,QoS-SSM
# --groups_start 5  --groups_stop 30  --groups_size_int 5
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups_parent_dirs", help="csv path dir containing group numbers' subdirs")
    parser.add_argument("--plot_save_dir", help="Location to save plots, default is local dir")
    parser.add_argument("--labels", help="experiments labels to for in order of groups_parent_dirs")
    parser.add_argument("--groups_start", help="starting number for multicast group size")
    parser.add_argument("--groups_stop", help="last number for multicast group size")
    parser.add_argument("--groups_size_int", help="interval for multicast group size")
    args = parser.parse_args()

    simulations_dirs = ''
    if args.groups_parent_dirs is not None and args.groups_start is not None \
            and args.groups_stop is not None and args.groups_size_int is not None \
            and args.labels is not None:
        labels = args.labels
        churn_congested_sim_dirs = args.groups_parent_dirs
        start = args.groups_start
        stop = args.groups_stop
        inter = args.groups_size_int
        run_list = range(1, 21, 1)
        plot_save_dir_c = args.plot_save_dir
    else:
        utils.working_dir = os.path.expanduser("/run/media/hsoni/TOSHIBA EXT/Review-exps-TNSM-l2bm/1g-churn-congested/")
        churn_congested_sim_dirs = utils.working_dir+'/churn-congested/dst,' \
                                   +utils.working_dir+'/churn-congested/dst-lb,' \
                                   +utils.working_dir+'/churn-congested/l2bm-10,' \
                                   +utils.working_dir+'/churn-congested/l2bm-40,' \
                                   +utils.working_dir+'/churn-congested/l2bm-60,'\
                                   +utils.working_dir+'/churn-congested/l2bm-20'

        # labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.2,L2BM-0.3,L2BM-0.4,L2BM-0.5,L2BM-0.6'
        labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.6,L2BM-0.4,L2BM-0.2'
        plot_save_dir_c = utils.working_dir+'/churn-congested/plots/'
        util.mkdir(plot_save_dir_c)
        osp_w_c = OtherStatsProcessorWChurn()
        start = 10
        stop = 150
        inter = 10
        run_list = range(500)
    color_dict = util.get_color_dict(churn_congested_sim_dirs, labels)
    marker_dict = util.get_marker_dict(churn_congested_sim_dirs, labels)
    do_mean_stats = True
    plot_link_util_metrics_for_different_groups(churn_congested_sim_dirs, plot_save_dir_c, labels, start, stop, inter,
                                                run_list, color_dict, marker_dict, do_mean_stats)
    osp_w_c.plot_other_metrics_for_different_groups(churn_congested_sim_dirs, plot_save_dir_c, labels,start, stop,
                                                    inter, run_list, color_dict, marker_dict)
    plot_time_sequence_based_metrics(churn_congested_sim_dirs, plot_save_dir_c, labels, 130, 190, 200, color_dict, marker_dict)
    plot_time_sequence_based_metrics(churn_congested_sim_dirs, plot_save_dir_c, labels, 130, 200, 200, color_dict, marker_dict)
    plot_time_sequence_based_metrics(churn_congested_sim_dirs, plot_save_dir_c, labels, 130, 10, 200, color_dict, marker_dict)
    group = 130
    runs = range(500)
    osp_w_c.plot_bw_acceptance_bar_graph(churn_congested_sim_dirs, plot_save_dir_c, labels, group, runs, color_dict)