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
import time
import argparse
import scipy.stats as ss
import matplotlib
import matplotlib.pyplot as plt
from Utils import *
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True

max_lu_th = 0.90

def main():
    b = [3, 5, 6, 4]
    nb = np.array(b)
    nbw = (nb > 4).sum()
    print nbw


# returns list
# {<link -> bw>}
def get_link_bw_stats_sim_res(link_stat_f):
    edge_consumption = []
    np_edge_consumption = np.empty
    try:
        with open(link_stat_f) as f:
            lines = [x.strip('\n') for x in f.readlines()]
        l = lines[0]
        edges_stats_l = l.strip(',').split(',(')
        for e in edges_stats_l:
            edge_stat = e.split(')=')
            # edge = edge_stat[0].strip('(')
            bw_cons = float(edge_stat[1].strip())
            edge_consumption.append((float(bw_cons)) / 100.0)
            np_edge_consumption = np.array(edge_consumption, dtype=float)
    except Exception as ex:
        print ('error in reading file named' + link_stat_f)
        print (ex)
        sys.exit(1)
    # print np_edge_consumption
    max = (np_edge_consumption >= max_lu_th ).sum()
    # max = np.amax(np_edge_consumption)
    return np.mean(np_edge_consumption), np.std(np_edge_consumption), (float(max)/102.0)*100.0


def find_greater_than_x(np_array, x):
    np.where(np_array > x)


def get_link_bw_stats_grid_res(run_loc):
    file_name = run_loc + "/link-bw-stats.txt"
    reciver_log_f = run_loc + "/receivers_count.log"
    ts_max_rec = get_all_receiver_timestamp(reciver_log_f)
    time_edge_consumption = get_group_bw_dict(file_name)
    return calculate_stddev_mean_sum_max_min_utilizedlink_for_ts(time_edge_consumption, ts_max_rec)


def calculate_stddev_mean_sum_max_min_utilizedlink_for_ts(time_edge_con, time_st):
    # timestamp_stddev_mean_sum_max_min_nouselink = []
    time_seq = 0
    count = 0
    # print(len(time_edge_con))
    for [ts, link_cons] in time_edge_con:
        # time.strptime(ts, "%H:%M:%S:+000")
        if ts > time_st:
            count += 1
            if count < 2:
                continue
            link_utils = []
            for cons in link_cons:
                link_util = (float(cons) / (1024*100))  # link consumption in Mbps
                # link_util = float(cons)
                # print(link_util)
                link_utils.append(link_util)
            np_link_utils = np.array(link_utils, dtype=float)
            # print (np_link_utils)
            if len(link_utils) == 0:
                continue
                # timestamp_std_mean_nouselink.append([time_seq, 0,0,0,0,0,0])
            else:
                time_seq = time_seq + 1
                max = (np_link_utils >= max_lu_th ).sum()
                return [np.mean(np_link_utils), np.std(np_link_utils), np.amax(np_link_utils)]
                # return [np.mean(np_link_utils), np.std(np_link_utils), (float(max)/102.0)*100.0]
    return []


def read_receiver_count_time_stamp_file(receiver_log):
    ts_receivers = []
    # tss = []
    try:
        with open(receiver_log) as f:
            lines = [x.strip('\n') for x in f.readlines()]
        for l in lines:
            line = l.strip().split('-')
            st = time.strptime(line[0], "%I:%M:%S +000")
            ts = time.mktime(st)
            ts_receivers.append([ts, line[1]])
    except Exception as ex:
        print ('error in reading file named ' + receiver_log)
        print (ex)
        sys.exit(1)
    return ts_receivers


def get_all_receiver_timestamp(receiver_log):
    ts_rec = read_receiver_count_time_stamp_file(receiver_log)
    [t, c] = ts_rec[-1]
    return t


def get_group_bw_dict(link_stat_f):
    time_edge_consumption = []
    try:
        with open(link_stat_f) as f:
            lines = [x.strip('\n') for x in f.readlines()]
        for l in lines:
            line = l.strip(',').strip('(')
            time_stat_l = line.split('-----')
            if len(time_stat_l) < 2 or time_stat_l[1] == '':
                continue
            # print str(time_stat_l)
            timestamp = time_stat_l[0].strip()
            timestamp = timestamp[:-4]
            ts = time.mktime(time.strptime(timestamp, '%I:%M:%S'))
            edges_stats_l = time_stat_l[1].strip(',( ').split(',(')
            # edges_stats_l =  re.split("{\(,\,\(}", time_stat_l[1].strip())
            used_edges = False
            edge_consumption = []
            # edge_consumption = {}
            for e in edges_stats_l:
                edge_stat = e.split(')=')
                edge = edge_stat[0].strip('(')
                bw_cons = int(edge_stat[1].strip())
                # edge_consumption[edge] = float(bw_cons)
                edge_consumption.append(float(bw_cons))
            time_edge_consumption.append([ts, edge_consumption])
    except Exception as ex:
        print ('error in reading file named ' + link_stat_f)
        print (ex)
        sys.exit(1)
    return time_edge_consumption



def get_single_run_lu_metrics(links_stat_dir, no_of_runs_dir, run_type='sim'):
    run_loc = links_stat_dir + "/run" + str(no_of_runs_dir)
    # total_tree_nodes, total_branch_nodes, receiver_requests_reject, bandwidth_reject = 1,1,1,1
    # bw_accept_map = {}
    if run_type == 'sim':
        file_name = links_stat_dir + "/run" + str(no_of_runs_dir) + "/link-bw-stats.txt"
        mean_link_util, std_link_util, max_link_util = get_link_bw_stats_sim_res(file_name)
    else:
        mean_link_util, std_link_util, max_link_util = get_link_bw_stats_grid_res(run_loc)
    return [mean_link_util, std_link_util, max_link_util]





def get_multi_runs_lu_metrics(links_stat_dir, no_of_runs_dir, run_type='sim'):
    multicast_link_run_metrics = []
    # multicast_bw_accept_map_metrics = {}
    for i in no_of_runs_dir:
        mean_std_max_link_util = get_single_run_lu_metrics(links_stat_dir, i, run_type=run_type)
        multicast_link_run_metrics.append(mean_std_max_link_util)
    np_multicast_link_node_run_metrics = np.array(multicast_link_run_metrics, dtype=float)
    # np_multicast_other_run_metrics = np.array(multicast_reject_run_metrics, dtype=float)
    avg_linkutil_dist_data = [np.mean(np_multicast_link_node_run_metrics[:, 0]),
                              np_multicast_link_node_run_metrics.shape[0],
                              ss.sem(np_multicast_link_node_run_metrics[:, 0])]
    std_dev_dist_data = [np.mean(np_multicast_link_node_run_metrics[:, 1]),
                         np_multicast_link_node_run_metrics.shape[0],
                         ss.sem(np_multicast_link_node_run_metrics[:, 1])]
    max_linkutil_dist_data = [np.mean(np_multicast_link_node_run_metrics[:, 2]),
                              np_multicast_link_node_run_metrics.shape[0],
                              ss.sem(np_multicast_link_node_run_metrics[:, 2])]
    return [avg_linkutil_dist_data, std_dev_dist_data, max_linkutil_dist_data]



def link_utils_for_different_groups(run_type, links_stat_dirs, plot_save_dir, labels, group_start,
                                    group_stop, group_int, group_runs):
    experiment_comp_paths = links_stat_dirs.split(',')
    ls = labels.split(',')
    algo_path_dict = {}
    i = 0
    for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
        algo_path_dict[lbl] = links_stat_dir
        i += 1
        # print links_stat_dir, lbl
    lbl_plot_data_dict = {}
    np_group_size = np.arange(int(group_start), int(group_stop) + 1, int(group_int))
    for lbl, links_stat_dir in algo_path_dict.iteritems():
        stddev_plot_data = []
        avg_util_plot_data = []
        max_util_plot_data = []
        for gs in np.nditer(np_group_size):
            dir = links_stat_dir + "/" + str(int(gs))
            [avg_linkutil_d, stddev_d, max_linkutil_d] = get_multi_runs_lu_metrics(dir, group_runs, run_type)
            stddev_plot_data.append(stddev_d)
            avg_util_plot_data.append(avg_linkutil_d)
            max_util_plot_data.append(max_linkutil_d)
        np_stddev_plot_data = np.array(stddev_plot_data)
        np_avg_util_plot_data = np.array(avg_util_plot_data)
        np_max_util_plot_data = np.array(max_util_plot_data)
        lbl_plot_data_dict[lbl] = [np_avg_util_plot_data, np_stddev_plot_data, np_max_util_plot_data]
    return lbl_plot_data_dict


def plot_link_util_metrics_for_different_groups(run_type, links_stat_dirs, plot_save_dir, labels, group_start,
                                                group_stop, group_int, group_runs, color_dict, marker_dict):

    lbl_plot_data_dict = link_utils_for_different_groups(run_type, links_stat_dirs, plot_save_dir, labels, group_start,
                                                         group_stop, group_int, group_runs)
    lbl_plot_data_dict_ls = [lbl_plot_data_dict]
    plot_average_data_points(lbl_plot_data_dict_ls, plot_save_dir, group_start, group_stop, group_int,
                             color_dict, marker_dict)
    # plot_bw_accept_bar_for_all_the_algos('bw-demands-accept', plot_save_dir, lbl_plot_data_dict,
    #                                      7,  color_dict, loc=None)


def plot_average_data_points(lbl_plot_data_dict_ls, plot_save_dir, group_start, group_stop, group_int,
                             color_dict, marker_dict):
    np_group_size = np.arange(int(group_start), int(group_stop) + 1, int(group_int))
    xticks = np.arange(int(group_start)-10, int(group_stop)+10 + 1, int(group_int))
    # yticks = np.arange(0.0, 1.06, 0.05)
    yticks = np.arange(0.15, 0.56, 0.05)
    loc = 'upper left'
    plot_metrics_for_all_the_algos('avg-lu', plot_save_dir, lbl_plot_data_dict_ls, 0, np_group_size,
                                   color_dict, marker_dict,
                                   'Number of Groups', 'Avg Link utilization',
                                   'Average link utilization Vs Number of groups', xticks_ls=xticks,
                                   ytick_ls=yticks, loc=loc)
    # yticks = np.arange(0.0, 1.06, 0.05)
    yticks = np.arange(0.0, 0.56, 0.05)
    plot_metrics_for_all_the_algos('stddev-lu', plot_save_dir, lbl_plot_data_dict_ls, 1, np_group_size,
                                   color_dict, marker_dict,
                                   'Number of Groups', 'StdDev Link utilization',
                                   'StdDev link utilization Vs Number of groups', xticks_ls=xticks,
                                   ytick_ls=yticks, loc=loc)
    loc = 'lower right'
    yticks = np.arange(0.40, 1.01, 0.05)
    plot_metrics_for_all_the_algos('max-lu', plot_save_dir, lbl_plot_data_dict_ls, 2, np_group_size,
                                   color_dict, marker_dict,
                                   'Number of Groups', 'Max Link utilization',
                                   'Max link utilization Vs Number of groups',
                                   xticks_ls=xticks, ytick_ls=yticks, loc=loc)



def plot_superimposed_datapoints_for_different_groups(links_stat_dirs_sims, links_stat_dirs_grid, plot_save_dir, labels,
                                                      group_start, group_stop, group_int, group_runs, color_dict):
    sim_lbl_plot_data_dict = link_utils_for_different_groups('sim', links_stat_dirs_sims, plot_save_dir,
                                                             labels, group_start, group_stop, group_int, group_runs)
    grid_lbl_plot_data_dict = link_utils_for_different_groups('grid', links_stat_dirs_grid, plot_save_dir,
                                                              labels, group_start, group_stop, group_int, group_runs)
    plot_average_data_points([sim_lbl_plot_data_dict, grid_lbl_plot_data_dict], plot_save_dir, group_start, group_stop,
                             group_int, color_dict, marker_dict)


def plot_metrics_for_all_the_algos(file_name, dest_dir_loca, lbl_plot_data_dict_ls, metrics_index, np_group_size,
                                   color_dict, marker_dict, xlable, ylable, title, x1=None, x2=None,
                                   xticks_ls=np.arange(40, 111, 10), y1=None, y2=None, ytick_ls=None, loc='upper left'):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    marker = itertools.cycle(('o', 'v', 's','*', 'd'))
    i = 0
    eb_l = []
    l_s = []
    if len(lbl_plot_data_dict_ls) == 1:
        linestyles = ['-']
    else:
        linestyles = ['--', '-']
    for lbl_plot_data_dict in lbl_plot_data_dict_ls:
        eb_l = []
        l_s = []
        sorted_order = sorted(lbl_plot_data_dict.keys())
        for l in sorted_order:
            plot_data = lbl_plot_data_dict[l]
            metric = plot_data[metrics_index]
            l_s.append(l)
            eb = plt.plot(np_group_size, metric[:,0], color=color_dict[l], marker=marker_dict[l], label=l)
            err = ss.t._ppf((1 + 0.95) / 2, np.subtract(metric[:, 1], 1))
            # print metric[:,1]
            # eb = plt.errorbar(np_group_size, metric[:, 0], color=color_dict[l], fmt=linestyles[i],
            #                   yerr=err * metric[:, 2], label=l)
            # eb[-1][0].set_linestyle(linestyles[i])
            eb_l.append(eb[0])
            plt.xlabel(xlable)
            plt.ylabel(ylable)
            plt.title(title)
            if xticks_ls is not None:
                plt.xticks(xticks_ls)
            elif x1 is not None and x2 is not None:
                plt.xlim(x1, x2)
            else:
                ''
            if ytick_ls is not None:
                plt.yticks(ytick_ls)
            elif y1 is not None and y2 is not None:
                plt.ylim(y1, y2)
            else:
                ''
        i += 1
    if i == 2:
        sim = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')
        testbed = plt.Line2D((0, 1), (0, 0), color='k', linestyle='-')
        eb_l += [sim, testbed]
        l_s += ['Simulation', 'Testbed']
    legends = ax.legend(eb_l, l_s, loc=loc, ncol=2, frameon=False)
    for t in legends.get_texts():
        if color_dict.has_key(t.get_text()):
            t.set_color(color_dict[t.get_text()])
        # print t.get_text()
    # fig.savefig(dest_dir_loca + file_name + ".svg", format='svg', dpi=1200)
    fig.savefig(dest_dir_loca + file_name + ".eps", format='eps', dpi=1200)




def compute_average_nomalized_datapoints_for_different_groups(run_type, links_stat_dirs, plot_save_dir, labels,
                                                              lbl_reference_model, group_start, group_stop, group_int,
                                                              group_runs, color_dict):
    experiment_comp_paths = links_stat_dirs.split(',')
    ls = labels.split(',')

    algo_path_dict = {}
    for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
        algo_path_dict[lbl] = links_stat_dir
        print links_stat_dir, lbl
    np_group_size = np.arange(int(group_start), int(group_stop) + 1, int(group_int))

    ref_alg_group_run_link_dict = {}
    np_ref_alg_group_run_link_dict = {}
    alg_group_run_link_dict = {}
    nrm_alg_group_run_link_dict = {}

    for gs in np.nditer(np_group_size):
        ref_alg_group_run_link_dict[int(gs)] = []
        for run_num in group_runs:
            ref_alg_path = algo_path_dict[lbl_reference_model]
            dir = ref_alg_path + "/" + str(int(gs))
            mean_std_max_link_util = get_single_run_lu_metrics(dir, run_num, run_type=run_type)
            ref_alg_group_run_link_dict[int(gs)].append(mean_std_max_link_util)
        np_ref_alg_group_run_link_dict[int(gs)] = np.array(ref_alg_group_run_link_dict[int(gs)], dtype=float)
        # print np.array(np_ref_alg_group_run_link_nodes_dict[int(gs)])

    algo_path_dict.pop(lbl_reference_model)

    for lbl, links_stat_dir in algo_path_dict.iteritems():
        for gs in np.nditer(np_group_size):
            alg_group_run_link_dict[(lbl, int(gs))] = []

    for gs in np.nditer(np_group_size):
        for run_num in group_runs:
            for lbl, links_stat_dir in algo_path_dict.iteritems():
                dir = links_stat_dir + "/" + str(int(gs))
                alg_group_link_run_data = get_single_run_lu_metrics(dir, run_num, run_type=run_type)
                alg_group_run_link_dict[(lbl, int(gs))].append(alg_group_link_run_data)

    for lbl, links_stat_dir in algo_path_dict.iteritems():
        for gs in np.nditer(np_group_size):
            np_algo_link_node_data = np.array(alg_group_run_link_dict[(lbl, int(gs))], dtype=float)
            ref_link_node_data = np_ref_alg_group_run_link_dict[int(gs)]
            nrm_alg_group_run_link_dict[(lbl, int(gs))] = np.true_divide(np.subtract(
                np_algo_link_node_data, ref_link_node_data), ref_link_node_data)
    lbl_plot_avg_data_dict = {}
    lbl_plot_std_data_dict = {}
    lbl_plot_max_data_dict = {}
    for lbl, links_stat_dir in algo_path_dict.iteritems():
        lbl_plot_avg_data_dict[lbl] = []
        lbl_plot_std_data_dict[lbl] = []
        lbl_plot_max_data_dict[lbl] = []
        for gs in np.nditer(np_group_size):
            # print lbl, links_stat_dir, nrm_alg_group_run_dict[(lbl,int(gs))].shape
            nrm_avg = nrm_alg_group_run_link_dict[(lbl, int(gs))][:, 0]
            nrm_std = nrm_alg_group_run_link_dict[(lbl, int(gs))][:, 1]
            nrm_max = nrm_alg_group_run_link_dict[(lbl, int(gs))][:, 2]
            # print lbl, gs, nrm_avg.shape[0]
            lbl_plot_avg_data_dict[lbl].append([np.mean(nrm_avg), ss.sem(nrm_avg), nrm_avg.shape[0]])
            lbl_plot_std_data_dict[lbl].append([np.mean(nrm_std), ss.sem(nrm_std), nrm_std.shape[0]])
            lbl_plot_max_data_dict[lbl].append([np.mean(nrm_max), ss.sem(nrm_max), nrm_max.shape[0]])

    np_lbl_plot_avg_data_dict = {}
    np_lbl_plot_std_data_dict = {}
    np_lbl_plot_max_data_dict = {}
    for lbl, links_stat_dir in algo_path_dict.iteritems():
        np_lbl_plot_avg_data_dict[lbl] = np.array(lbl_plot_avg_data_dict[lbl])
        np_lbl_plot_std_data_dict[lbl] = np.array(lbl_plot_std_data_dict[lbl])
        np_lbl_plot_max_data_dict[lbl] = np.array(lbl_plot_max_data_dict[lbl])
        # print lbl, np_lbl_plot_avg_data_dict[lbl].shape
        # print lbl, np_lbl_plot_std_data_dict[lbl].shape
        # print lbl, np_lbl_plot_max_data_dict[lbl].shape

    return np_lbl_plot_avg_data_dict, np_lbl_plot_std_data_dict, np_lbl_plot_max_data_dict



def plot_average_nomalized_datapoints_for_different_groups(run_type, links_stat_dirs, plot_save_dir, labels,
                                                           group_start, group_stop, group_int, group_runs,
                                                           color_dict, marker_dict):
    ref_label = 'DST-PL'
    np_lbl_plot_avg_data_dict, np_lbl_plot_std_data_dict, np_lbl_plot_max_data_dict = \
        compute_average_nomalized_datapoints_for_different_groups(run_type, links_stat_dirs, plot_save_dir, labels,
                                                                  ref_label, group_start, group_stop, group_int,
                                                                  group_runs, color_dict)
    np_group_size = np.arange(int(group_start), int(group_stop) + 1, int(group_int))
    plot_normalized_data(np_group_size, color_dict, marker_dict, ref_label, np_lbl_plot_avg_data_dict, np_lbl_plot_std_data_dict,
                         np_lbl_plot_max_data_dict)


def plot_normalized_data(np_group_size, color_dict, marker_dict, lbl_reference_model, np_lbl_plot_avg_data_dict_ls,
                         np_lbl_plot_std_data_dict_ls,
                         np_lbl_plot_max_data_dict_ls):
    loc = 'upper left'
    xlable = 'Number of Groups'
    yticks = np.arange(-0.10, 0.26, 0.05)
    # xticks = np.arange(20, 81, 10)
    xticks = np.arange(40, 111, 10)
    # yticks = np.arange(0.0, 0.30, 0.05)
    plot_normalized_metrics_for_all_the_algos('norm-avg-lu', plot_save_dir, np_lbl_plot_avg_data_dict_ls, None,
                                              np_group_size, color_dict, marker_dict, xlable,
                                              'Normalized Average Link utilization',
                                              'Normalized Average link utilization Vs Number of groups',
                                              xticks_ls=xticks,
                                              yticks_ls=yticks, loc=loc, reference_skip=lbl_reference_model)
    # yticks = np.arange(-0.45, 0.15, 0.05)
    loc = 'upper left'
    yticks = np.arange(-0.5, 0.06, 0.05)
    plot_normalized_metrics_for_all_the_algos('norm-max-lu', plot_save_dir, np_lbl_plot_max_data_dict_ls, None,
                                              np_group_size, color_dict, marker_dict, xlable,
                                              'Normalized Max Link utilization',
                                              'Normalized Max Link utilization Vs Number of groups', xticks_ls=xticks,
                                              yticks_ls=yticks, loc=loc, reference_skip=lbl_reference_model)
    yticks = np.arange(-0.5, -0.04, 0.05)
    # yticks = np.arange(-0.5, 0.2, 0.05)
    loc = 'upper right'
    plot_normalized_metrics_for_all_the_algos('norm-stddev-lu', plot_save_dir, np_lbl_plot_std_data_dict_ls, None,
                                              np_group_size, color_dict, marker_dict, xlable,
                                              'Normalized Stddev utilization',
                                              'Normalized Stddev utilization Vs Number of groups', xticks_ls=xticks,
                                              yticks_ls=yticks, loc=loc, reference_skip=lbl_reference_model)


def plot_superimposed_nomalized_datapoints_for_different_groups(links_stat_dirs_sims, links_stat_dirs_grid,
                                                                plot_save_dir,
                                                                labels, group_start, group_stop, group_int, group_runs,
                                                                color_dict=None):
    ref_label = 'DST-PL'
    np_lbl_plot_avg_data_sim_dict, np_lbl_plot_std_data_sim_dict, \
    np_lbl_plot_max_data_sim_dict = compute_average_nomalized_datapoints_for_different_groups(
        'sim', links_stat_dirs_sims, plot_save_dir, labels, ref_label, group_start, group_stop, group_int,
        group_runs, color_dict)

    np_lbl_plot_avg_data_grid_dict, np_lbl_plot_std_data_grid_dict, np_lbl_plot_max_data_grid_dict = \
        compute_average_nomalized_datapoints_for_different_groups(
            'grid', links_stat_dirs_grid, plot_save_dir, labels, ref_label, group_start, group_stop, group_int,
            group_runs, color_dict)
    np_group_size = np.arange(int(group_start), int(group_stop) + 1, int(group_int))
    plot_normalized_data(np_group_size, color_dict, ref_label,
                         [np_lbl_plot_avg_data_sim_dict, np_lbl_plot_avg_data_grid_dict],
                         [np_lbl_plot_max_data_sim_dict, np_lbl_plot_max_data_grid_dict],
                         [np_lbl_plot_std_data_sim_dict, np_lbl_plot_std_data_grid_dict])



def plot_normalized_metrics_for_all_the_algos(file_name, dest_dir, lbl_plot_dict, metrics_index, np_group_size,
                                              colors, marker_dict, xlable, ylable, title, x1=None, x2=None,
                                              xticks_ls=None, y1=None, y2=None,
                                              yticks_ls=None, loc='upper left', reference_skip=None):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    marker = itertools.cycle(('o-', 'v-', 's-','*-', 'd-'))
    # marker = itertools.cycle(('v', 's', '8', 'o', 'p', 'D'))
    mark = marker.next()
    i = 0
    eb_l = []
    l_s = []
    for lbl, data in lbl_plot_dict.iteritems():
        l = lbl + '-Vs-' + reference_skip
        l_s.append(l)
        err = ss.t._ppf((1 + 0.95) / 2, np.subtract(data[:, 2], 1))
        eb = plt.errorbar(np_group_size, data[:, 0], color=colors[lbl], fmt=marker_dict[lbl]+'-', yerr=err * data[:, 1],
                          label=l)
        eb_l.append(eb)
        # plt.plot(np_group_size, data[:,0], color=colors[lbl], marker=mark, label=l)
        # plt.legend(loc=loc, frameon=False)
        plt.xlabel(xlable)
        plt.ylabel(ylable)
        plt.title(title)
        if xticks_ls is not None:
            plt.xticks(xticks_ls)
        elif x1 is not None and x2 is not None:
            plt.xlim(x1, x2)
        else:
            ''
        if yticks_ls is not None:
            plt.yticks(yticks_ls)
        elif y1 is not None and y2 is not None:
            plt.ylim(y1, y2)
        else:
            ''
        # i += 1
    if i == 2:
        sim = plt.Line2D((0, 1), (0, 0), color='k', linestyle='-')
        testbed = plt.Line2D((0, 1), (0, 0), color='k', linestyle='--')
        eb_l += [sim, testbed]
        l_s += ['Simulation', 'Testbed']
    ax.legend(eb_l, l_s, ncol=2, frameon=False)
    # fig.savefig(dest_dir + file_name + ".svg", format='svg', dpi=1200)
    fig.savefig(dest_dir + file_name + ".eps", format='eps', dpi=1200)



def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


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
        groups_parent_dirs_grid = utils.working_dir+'/qos-multicast-compile/exp2-2mb-10-100/dst,' \
                                  +utils.working_dir+'/qos-multicast-compile/exp2-2mb-10-100/lb,' \
                                  +utils.working_dir+'/qos-multicast-compile/exp2-2mb-50-100-LLDMs/dcbr-10-10,' \
                                  +utils.working_dir+'/qos-multicast-compile/exp2-2mb-50-100-LLDMs/dcbr-40-10,' \
                                  +utils.working_dir+'/qos-multicast-compile/exp2-2mb-50-100-LLDMs/dcbr-60-10'
        # groups_parent_dirs_sim = utils.working_dir+'/qos-multicast-compile/sim-exp2-2mb-10-100/dst,' \
        #                          +utils.working_dir+'/qos-multicast-compile/sim-exp2-2mb-10-100/dst-lb,' \
        #                          +utils.working_dir+'/qos-multicast-compile/sim-exp2-2mb-10-100/l2bm-10,' \
        #                          +utils.working_dir+'/qos-multicast-compile/sim-exp2-2mb-10-100/l2bm-40,' \
        #                          +utils.working_dir+'/qos-multicast-compile/sim-exp2-2mb-10-100/l2bm-60'
        # groups_parent_dirs_sim = utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/dst,' \
        #                          +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/dst-lb,' \
        #                          +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/l2bm-10,' \
        #                          +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/l2bm-40,' \
        #                          +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/l2bm-60'
        # groups_parent_dirs_grid = utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-75/dst,' \
        #                           +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-75/lb,' \
        #                           +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-10-10,' \
        #                           +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-40-10,' \
        #                           +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-60-10'
        # simulations_dirs = utils.working_dir+'/qos-multicast-compile/simulation/wo-churn-congested-va-bw/dst,' \
        #                    +utils.working_dir+'/qos-multicast-compile/simulation/wo-churn-congested-va-bw/dst-lb,' \
        #                    +utils.working_dir+'/qos-multicast-compile/simulation/wo-churn-congested-va-bw/l2bm-10,' \
        #                    +utils.working_dir+'/qos-multicast-compile/simulation/wo-churn-congested-va-bw/l2bm-40,' \
        #                    +utils.working_dir+'/qos-multicast-compile/simulation/wo-churn-congested-va-bw/l2bm-60'
        labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.4,L2BM-0.6'
        start = 50
        stop = 100
        inter = 10
        run_list = range(1, 21, 1)
        # run_list = range(500)
    plot_save_dir = utils.working_dir+'/qos-multicast-compile/exp2-2mb-10-100/plots/'
    mkdir(plot_save_dir)
    color_dict = utils.get_color_dict(groups_parent_dirs_grid, labels)
    marker_dict = utils.get_marker_dict(groups_parent_dirs_grid, labels)
    plot_link_util_metrics_for_different_groups('grid', groups_parent_dirs_grid, plot_save_dir, labels,
                                                        start, stop, inter, run_list, color_dict, marker_dict)
    # plot_superimposed_datapoints_for_different_groups(groups_parent_dirs_sim, groups_parent_dirs_grid,
    #                                                   plot_save_dir, labels, start, stop, inter, run_list,
    #                                                   color_dict)
    plot_average_nomalized_datapoints_for_different_groups('grid', groups_parent_dirs_grid, plot_save_dir, labels,
                                                           start, stop, inter, run_list, color_dict, marker_dict)
    # plot_superimposed_nomalized_datapoints_for_different_groups(groups_parent_dirs_sim, groups_parent_dirs_grid,
    #                                                             plot_save_dir, labels, start, stop, inter, run_list,
    #                                                             color_dict)
