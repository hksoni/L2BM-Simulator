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
import json
import sys
import  Heap
import argparse
import matplotlib
matplotlib.use('Agg')
from Utils import *
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
# matplotlib.rcParams['text.usetex'] = True
import scipy.stats as ss
import numpy as np
import Utils  as util
import itertools
import matplotlib.pyplot as plt
plt.style.use('classic')
import Utils as utils

def main():
    b = [3, 5, 6, 4]
    nb = np.array(b)
    nbw = (nb > 4).sum()
    print nbw


class OtherStatsProcessorWOChurn(object):

    def __init__(self):
        print ''

    def get_other_dict_stats(self, links_stat_dir, run_num):
        other_stat_f = links_stat_dir+"/run" + str(run_num)+"/other-results.txt"
        try:
            with open(other_stat_f) as f:
                lines = [x.strip('\n') for x in f.readlines()]
            bw_accept_map = eval(lines[4].split('=')[1])
            bw_accept_ratio_map = eval(lines[7].split('=')[1])
            bw_accept_ratio_map = {int(k):float(v) for k,v in bw_accept_ratio_map.items()}
            bw_accept_map = {int(k):int(v) for k,v in bw_accept_map.items()}
        except Exception as ex:
            print ('Exception in get_other_stats_wo_churn: ' + other_stat_f)
            print (ex)
            sys.exit(1)
        return bw_accept_map, bw_accept_ratio_map


    def get_other_stats(self, links_stat_dir, run_num):
        other_stat_f = links_stat_dir+"/run" + str(run_num)+"/other-results.txt"
        try:
            with open(other_stat_f) as f:
                lines = [x.strip('\n') for x in f.readlines()]
            receiver_requests_accept = lines[0].split('=')[1]
            bandwidth_accept = lines[1].split('=')[1]
            branch_nodes_ll = lines[2].split('=')[1].replace('\'', '"')
            l = json.loads(branch_nodes_ll)
            lens = [len(x) for x in l]
            total_branch_nodes = sum(lens)
            total_tree_nodes = lines[3].split('=')[1]
            receiver_accept_ratio = lines[5].split('=')[1]
            bandwidth_accept_ratio = lines[6].split('=')[1]
        except Exception as ex:
            print ('Exception in get_other_stats_wo_churn: ' + other_stat_f)
            print (ex)
            sys.exit(1)
        return total_tree_nodes, total_branch_nodes, receiver_requests_accept, bandwidth_accept, \
               receiver_accept_ratio, bandwidth_accept_ratio


    def get_single_run_other_metrics(self, links_stat_dir, run_num):
        total_tree_nodes, total_branch_nodes, receiver_requests_accept, \
        bandwidth_accept, receiver_accept_ratio, bandwidth_accept_ratio = self.get_other_stats(links_stat_dir, run_num)
        return [total_tree_nodes, total_branch_nodes, receiver_requests_accept, bandwidth_accept,
                receiver_accept_ratio, bandwidth_accept_ratio]


    def get_multi_runs_other_dict_metrics(self, links_stat_dir, no_of_runs_dir):
        multicast_bw_accept_map_metrics = {}
        multicast_bw_accept_ratio_map_metrics = {}
        for i in no_of_runs_dir:
            bw_accept_map, bw_accept_ratio_map = self.get_other_dict_stats(links_stat_dir, i)
            for k,v in bw_accept_map.iteritems():
                if multicast_bw_accept_map_metrics.has_key(k):
                    multicast_bw_accept_map_metrics[k].append(v)
                else:
                    multicast_bw_accept_map_metrics[k] = [v]
            for k,v in bw_accept_ratio_map.iteritems():
                if multicast_bw_accept_ratio_map_metrics.has_key(k):
                    multicast_bw_accept_ratio_map_metrics[k].append(v)
                else:
                    multicast_bw_accept_ratio_map_metrics[k] = [v]
        multicast_bw_accept_map_data = {}
        multicast_bw_accept_ratio_map_data = {}
        for k,v in multicast_bw_accept_map_metrics.iteritems():
            np_array = np.array(v, dtype=float)
            if np_array.shape[0] == 1:
                err = np.std(np_array)
            else:
                err = ss.sem(np_array)
            multicast_bw_accept_map_data[k] = [np.mean(np_array), np_array.shape[0], err]
        for k,v in multicast_bw_accept_ratio_map_metrics.iteritems():
            np_array = np.array(v, dtype=float)
            if np_array.shape[0] == 1:
                err = np.std(np_array)
            else:
                err = ss.sem(np_array)
            multicast_bw_accept_ratio_map_data[k] = [np.mean(np_array), np_array.shape[0], err]
        return multicast_bw_accept_map_data, multicast_bw_accept_ratio_map_data



    def get_multi_runs_other_metrics(self, links_stat_dir, no_of_runs_dir):
        multicast_other_run_metrics = []
        for i in no_of_runs_dir:
            other_metrics = self.get_single_run_other_metrics(links_stat_dir, i)
            multicast_other_run_metrics.append(other_metrics)
        np_multicast_other_run_metrics = np.array(multicast_other_run_metrics, dtype=float)
        total_tree_dist_data = [np.mean(np_multicast_other_run_metrics[:,0]),
                                np_multicast_other_run_metrics.shape[0],
                                ss.sem(np_multicast_other_run_metrics[:,0])]
        branch_nodes_dist_data = [np.mean(np_multicast_other_run_metrics[:,1]),
                                  np_multicast_other_run_metrics.shape[0],
                                  ss.sem(np_multicast_other_run_metrics[:,1])]
        receiver_accept_dist_data = [np.mean(np_multicast_other_run_metrics[:,2]),
                                     np_multicast_other_run_metrics.shape[0],
                                     ss.sem(np_multicast_other_run_metrics[:,2])]
        bw_accept_dist_data = [np.mean(np_multicast_other_run_metrics[:,3]),
                               np_multicast_other_run_metrics.shape[0],
                               ss.sem(np_multicast_other_run_metrics[:,3])]
        receiver_accept_ratio_dist_data = [np.mean(np_multicast_other_run_metrics[:,4]),
                                     np_multicast_other_run_metrics.shape[0],
                                     ss.sem(np_multicast_other_run_metrics[:,4])]
        bw_accept_ratio_dist_data = [np.mean(np_multicast_other_run_metrics[:,5]),
                               np_multicast_other_run_metrics.shape[0],
                               ss.sem(np_multicast_other_run_metrics[:,5])]
        return [total_tree_dist_data, branch_nodes_dist_data, receiver_accept_dist_data,
                bw_accept_dist_data, receiver_accept_ratio_dist_data, bw_accept_ratio_dist_data]


    def get_other_metrics_for_different_groups(self, links_stat_dirs, labels, group_start, group_stop,
                                               group_int, group_runs):
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
            receiver_accept_plot_data = []
            bw_accept_plot_data = []
            receiver_accept_ratio_plot_data = []
            bw_accept_plot_ratio_data = []
            branch_nodes_plot_data = []
            total_tree_plot_data = []
            for gs in np.nditer(np_group_size):
                dir = links_stat_dir + "/" + str(int(gs))
                [total_tree_d, branch_nodes_d, receiver_accept_d, bw_accept_d,
                 receiver_accept_ratio_d, bw_accept_ratio_d] = self.get_multi_runs_other_metrics(dir, group_runs)
                receiver_accept_plot_data.append(receiver_accept_d)
                bw_accept_plot_data.append(bw_accept_d)
                branch_nodes_plot_data.append(branch_nodes_d)
                total_tree_plot_data.append(total_tree_d)
                receiver_accept_ratio_plot_data.append(receiver_accept_ratio_d)
                bw_accept_plot_ratio_data.append(bw_accept_ratio_d)
            np_receiver_accept_plot_data = np.array(receiver_accept_plot_data, dtype=float)
            np_bw_accept_plot_data = np.array(bw_accept_plot_data, dtype=float)
            np_receiver_accept_ratio_plot_data = np.array(receiver_accept_ratio_plot_data, dtype=float)
            np_bw_accept_plot_ratio_data = np.array(bw_accept_plot_ratio_data, dtype=float)
            np_branch_nodes_plot_data = np.array(branch_nodes_plot_data, dtype=float)
            np_total_tree_plot_data = np.array(total_tree_plot_data, dtype=float)
            lbl_plot_data_dict[lbl] = [np_receiver_accept_plot_data, np_bw_accept_plot_data,
                                       np_branch_nodes_plot_data, np_total_tree_plot_data,
                                       np_receiver_accept_ratio_plot_data, np_bw_accept_plot_ratio_data]
        return lbl_plot_data_dict



    def plot_other_metrics_data_points(self, lbl_plot_data_dict_ls, plot_save_dir, group_start, group_stop,
                                       group_int, color_dict, marker_dict):
        np_group_size = np.arange(int(group_start), int(group_stop) + 1, int(group_int))
        xticks = np.arange(int(group_start)-int(group_int), int(group_stop)+int(group_int) + 1, int(group_int))
        exp = int(np.log10(group_int))
        if exp == 2:
            lbl_prefix = "(in hundreds)"
        elif exp == 3:
            lbl_prefix = "(in thousands)"
        elif exp == 4:
            lbl_prefix = "(in ten thousands)"
        else:
            exp = 0
            lbl_prefix = ''
        xticks = np.true_divide(xticks, 10**exp)
        np_group_size = np.true_divide(np_group_size, 10**exp)
        xlabel = 'Number of Groups' + lbl_prefix
        plot_metrics_for_all_the_algos('recv-accept', plot_save_dir, lbl_plot_data_dict_ls, 0, np_group_size, color_dict,
                                       marker_dict, xlabel, 'Number of Join Request Accepted',
                                       'Join Request Accepted Vs Number of groups', xticks_ls=xticks, loc=0)
        plot_metrics_for_all_the_algos('bw-accept', plot_save_dir, lbl_plot_data_dict_ls, 1, np_group_size, color_dict,
                                       marker_dict, xlabel, 'Bandwidth Demands',
                                       'Bandwidth Demands Vs Number of groups', xticks_ls=xticks, loc=0)
        plot_metrics_for_all_the_algos('total-branch-nodes', plot_save_dir, lbl_plot_data_dict_ls, 2, np_group_size,
                                       color_dict, marker_dict, xlabel, 'Total Branch Nodes',
                                       'Total Branch Nodes Vs Number of groups', xticks_ls=xticks, loc=0)
        plot_metrics_for_all_the_algos('total-tree-nodes', plot_save_dir, lbl_plot_data_dict_ls, 3, np_group_size,
                                       color_dict, marker_dict, xlabel, 'Total Tree Nodes',
                                       'Total Tree Nodes Vs Number of groups', xticks_ls=xticks, loc=0)
        yticks = np.arange(0.4, 1.11, 0.05)
        plot_metrics_for_all_the_algos('recv-accept-ratio', plot_save_dir, lbl_plot_data_dict_ls, 4, np_group_size,
                                       color_dict, marker_dict, xlabel, 'Multicast Joins Acceptance Ratio',
                                       'Multicast Joins Acceptance Ratio Vs Number of groups', xticks_ls=xticks,
                                       ytick_ls=yticks, loc=0)
        plot_metrics_for_all_the_algos('bw-accept-ratio', plot_save_dir, lbl_plot_data_dict_ls, 5, np_group_size,
                                       color_dict, marker_dict, xlabel, 'BW. Demands Acceptance Ratio',
                                       'BW. Demands Acceptance Ratio Vs Number of groups', xticks_ls=xticks,
                                       ytick_ls=yticks, loc=0)


    def plot_other_metrics_for_different_groups(self, links_stat_dirs, plot_save_dir, labels, group_start, group_stop,
                                                group_int, group_runs, color_dict, marker_dict):
        lbl_plot_data_dict = self.get_other_metrics_for_different_groups(links_stat_dirs, labels, group_start,
                                                                         group_stop, group_int, group_runs)
        lbl_plot_data_dict_ls = [lbl_plot_data_dict]
        self.plot_other_metrics_data_points(lbl_plot_data_dict_ls, plot_save_dir, group_start, group_stop,
                                            group_int, color_dict, marker_dict)


    def plot_bw_acceptance_bar_graph(self, links_stat_dirs, plot_save_dir, labels, group, group_runs, color_dict):
        lbl_plot_data_dict = self.get_bw_acceptance_bar_graph_data(links_stat_dirs, labels, group, group_runs)
        # file_name = 'bw-accept-'+str(group)
        title = 'Accepatance Ratios for different BW. Demands'
        # self.plot_bw_accept_bar_for_all_the_algos(file_name, plot_save_dir, lbl_plot_data_dict, 0, color_dict, ylabel)
        file_name = 'bw-accept-ratio-'+str(group)
        ylabel = 'Join Request Accepatance Ratio'
        yticks = np.arange(0.0, 1.3, 0.1)
        self.plot_bw_accept_bar_for_all_the_algos(file_name, plot_save_dir, lbl_plot_data_dict, 1, color_dict,
                                                  title, ylabel, yticks=yticks)


    def get_bw_acceptance_bar_graph_data(self, links_stat_dirs, labels, group, group_runs):
        experiment_comp_paths = links_stat_dirs.split(',')
        ls = labels.split(',')
        algo_path_dict = {}
        for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
            algo_path_dict[lbl] = links_stat_dir
            # print links_stat_dir, lbl
        lbl_plot_data_dict = {}
        for lbl, links_stat_dir in algo_path_dict.iteritems():
            dir = links_stat_dir + "/" + str(int(group))
            multicast_bw_accept_map_data, multicast_bw_accept_ratio_map_data \
                = self.get_multi_runs_other_dict_metrics(dir, group_runs)
            lbl_plot_data_dict[lbl] = [multicast_bw_accept_map_data, multicast_bw_accept_ratio_map_data ]
        return lbl_plot_data_dict


    def plot_bw_accept_bar_for_all_the_algos(self, file_name, dest_dir_loca, lbl_plot_data_dict, metric_index,
                                             color_dict, title, ylable, yticks=None, loc=None):
        print 'plot_bw_accept_bar_for_all_the_algos'
        plt.clf()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        width = 0.1
        ll = ''
        keys = []
        rects_ls = []
        labels = []
        i = 0
        sorted_order = sorted(lbl_plot_data_dict.keys())
        for l in sorted_order:
            plot_data = lbl_plot_data_dict[l]
        # for l, plot_data in lbl_plot_data_dict.iteritems():
            bw_demands_map = plot_data[metric_index]
            heap = Heap.Heap()
            for k, v in bw_demands_map.iteritems():
                heap.push(k, k, v)
            # print l
            # print bw_demands_map
            current_keys = []
            mean = []
            std_dev = []
            s = 2
            while not heap.is_empty():
                k, key, v = heap.pop()
            # for k, v in bw_demands_map.iteritems():
                current_keys.append(k)
                mean.append(v[0])
                s = v[1]
                std_dev.append(v[2])
            if keys != [] and keys != current_keys:
                print 'Bandwidth keys are not identical across lables'
                print 'Label : ' + ll + ' has ' + str(keys)
                print 'Label : ' + l + ' has ' + str(current_keys)
                print 'exiting'
                return
            err = ss.t._ppf((1 + 0.95) / 2, np.subtract(s, 1))
            yerr = err * np.array(std_dev)
            # print mean
            # print yerr
            ind = np.arange(len(current_keys)) + (i * width)
            rect = ax.bar(ind, mean, width, color=color_dict[l], yerr=yerr)
            rects_ls.append(rect)
            labels.append(l)
            i += 1
            keys = current_keys
            ll = l
        ax.set_ylabel(ylable)
        ax.set_xlabel('Multicast Group Bandwidth Demands')
        ax.set_title(title)

        ax.set_xticks(ind - width)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.set_xticklabels(keys)
        legends = ax.legend(rects_ls, labels, ncol=4, frameon=False, loc=0, prop={'size':10})

        def autolabel(rects):
            # attach some text labels
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height,
                        '%d' % int(height),
                        ha='center', va='bottom', fontsize=8)
        # for r in rects_ls:
        #     autolabel(r)
        for t in legends.get_texts():
            if color_dict.has_key(t.get_text()):
                t.set_color(color_dict[t.get_text()])
        # fig.savefig(dest_dir_loca + file_name + ".svg", format='svg', dpi=1200)
        fig.savefig(dest_dir_loca + file_name + ".eps", format='eps', dpi=1200)






def plot_metrics_for_all_the_algos(file_name, dest_dir_loca, lbl_plot_data_dict_ls, metrics_index, np_group_size,
                                   color_dict, marker_dict, xlable, ylable, title, x1=None, x2=None,
                                   xticks_ls=np.arange(40, 111, 10), y1=None, y2=None, ytick_ls=None,
                                   loc=0):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
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
        # for l, plot_data in lbl_plot_data_dict.items():
            metric = plot_data[metrics_index]
            l_s.append(l)
            # plt.plot(np_group_size, metric[:,0], color=color_dict[l], marker=marker.next(), label=l)
            err = ss.t._ppf((1 + 0.95) / 2, np.subtract(metric[:, 1], 1))
            # print metric[:,1]
            eb = plt.errorbar(np_group_size, metric[:, 0], color=color_dict[l], fmt=marker_dict[l]+linestyles[i],
                              yerr=err * metric[:, 2], label=l)
            eb[-1][0].set_linestyle(linestyles[i])
            eb_l.append(eb)
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
    fig.savefig(dest_dir_loca + file_name + ".svg", format='svg', dpi=1200)
    fig.savefig(dest_dir_loca + file_name + ".eps", format='eps', dpi=1200)


class OtherStatsProcessorWChurn(OtherStatsProcessorWOChurn):

    def get_other_dict_stats(self, links_stat_dir, run_num):
        other_stat_f = links_stat_dir+"/run" + str(run_num)+"/churn-other-results.txt"
        try:
            with open(other_stat_f) as f:
                lines = [x.strip('\n') for x in f.readlines()]
            bw_accept_map = eval(lines[2].split('=')[1])
            bw_accept_ratio_map = eval(lines[5].split('=')[1])
            bw_accept_ratio_map = {int(k):float(v) for k,v in bw_accept_ratio_map.items()}
            bw_accept_map = {int(k):int(v) for k,v in bw_accept_map.items()}
        except Exception as ex:
            print ('Exception in get_other_stats_wo_churn: ' + other_stat_f)
            print (ex)
            sys.exit(1)
        return bw_accept_map, bw_accept_ratio_map


    def get_other_stats(self, links_stat_dir, run_num):

        other_stat_f = links_stat_dir+"/run" + str(run_num)+"/churn-other-results.txt"
        try:
            with open(other_stat_f) as f:
                lines = [x.strip('\n') for x in f.readlines()]
            receiver_requests_accept = lines[0].split('=')[1]
            bandwidth_accept = lines[1].split('=')[1]
            receiver_accept_ratio = lines[3].split('=')[1]
            bandwidth_accept_ratio = lines[4].split('=')[1]
        except Exception as ex:
            print ('Exception in get_other_stats_wo_churn: ' + other_stat_f)
            print (ex)
            sys.exit(1)
        return 0, 0, receiver_requests_accept, bandwidth_accept, \
               receiver_accept_ratio, bandwidth_accept_ratio


class OtherStatsProcessorWOChurnTestbed(OtherStatsProcessorWOChurn):

    def __init__(self, ip_bw_map_file):
        super(OtherStatsProcessorWOChurn, self).__init__()
        self.group_ip_bw_map_dict = util.get_group_ip_bw_dict(ip_bw_map_file)
        self.group_bw_map_dict = util.get_group_bw_dict(ip_bw_map_file)


    def get_other_stats(self, links_stat_dir, run_num):
        other_stat_f = links_stat_dir+"/run" + str(run_num)+"/other-results.txt"
        total_tree_nodes = 1
        total_branch_nodes = 1
        receiver_requests_accept = 1
        bandwidth_accept = 0
        receiver_accept_ratio = 1
        bandwidth_accept_ratio = 1

        try:
            with open(other_stat_f) as f:
                lines = [x.strip('\n') for x in f.readlines()]
            for line in lines:
                bandwidth_accept += float(self.group_ip_bw_map_dict[line])
        except Exception as ex:
            print ('Exception in get_other_stats_wo_churn: ' + other_stat_f)
            print (ex)
            sys.exit(1)
        return total_tree_nodes, total_branch_nodes, receiver_requests_accept, bandwidth_accept, \
               receiver_accept_ratio, bandwidth_accept_ratio


    def get_total_bw_demands(self, no_groups, receivers_per_group=13.0):
        total_demands = 0.0
        for i in range(0,no_groups):
            total_demands += (receivers_per_group * float(self.group_bw_map_dict[i][1]))
        return total_demands


    def get_other_metrics_for_different_groups(self, links_stat_dirs, labels, group_start, group_stop,
                                               group_int, group_runs):
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
            receiver_accept_plot_data = []
            bw_accept_plot_data = []
            receiver_accept_ratio_plot_data = []
            bw_accept_plot_ratio_data = []
            branch_nodes_plot_data = []
            total_tree_plot_data = []
            for gs in np.nditer(np_group_size):
                total_bw_demands = self.get_total_bw_demands(gs)
                dir = links_stat_dir + "/" + str(int(gs))
                [total_tree_d, branch_nodes_d, receiver_accept_d, bw_accept_d,
                 receiver_accept_ratio_d, bw_accept_ratio_d] = self.get_multi_runs_other_metrics(dir, group_runs)
                bw_accept_ratio_d = bw_accept_d
                bw_accept_ratio_d[0] /= total_bw_demands
                bw_accept_ratio_d[2] /= total_bw_demands
                receiver_accept_plot_data.append(receiver_accept_d)
                bw_accept_plot_data.append(bw_accept_d)
                branch_nodes_plot_data.append(branch_nodes_d)
                total_tree_plot_data.append(total_tree_d)
                receiver_accept_ratio_plot_data.append(receiver_accept_ratio_d)
                bw_accept_plot_ratio_data.append(bw_accept_ratio_d)
            np_receiver_accept_plot_data = np.array(receiver_accept_plot_data, dtype=float)
            np_bw_accept_plot_data = np.array(bw_accept_plot_data, dtype=float)
            np_receiver_accept_ratio_plot_data = np.array(receiver_accept_ratio_plot_data, dtype=float)
            np_bw_accept_plot_ratio_data = np.array(bw_accept_plot_ratio_data, dtype=float)
            np_branch_nodes_plot_data = np.array(branch_nodes_plot_data, dtype=float)
            np_total_tree_plot_data = np.array(total_tree_plot_data, dtype=float)
            lbl_plot_data_dict[lbl] = [np_receiver_accept_plot_data, np_bw_accept_plot_data,
                                       np_branch_nodes_plot_data, np_total_tree_plot_data,
                                       np_receiver_accept_ratio_plot_data, np_bw_accept_plot_ratio_data]
        return lbl_plot_data_dict




def plot_superimposed_bw_accept_ratio_for_different_groups(links_stat_dirs_sims, links_stat_dirs_grid, plot_save_dir,
                                                           labels, group_start, group_stop, group_int, group_runs,
                                                           color_dict, marker_dict):
    osp_wo_c_sim = OtherStatsProcessorWOChurn()
    osp_wo_c_grid = OtherStatsProcessorWOChurnTestbed(util.bw_map_file)

    sim_lbl_plot_data_dict = osp_wo_c_sim.get_other_metrics_for_different_groups(
        links_stat_dirs_sims, labels, group_start, group_stop, group_int, group_runs)
    grid_lbl_plot_data_dict = osp_wo_c_grid.get_other_metrics_for_different_groups(
        links_stat_dirs_grid,  labels, group_start, group_stop, group_int, group_runs)
    lbl_plot_data_dict_ls = [sim_lbl_plot_data_dict, grid_lbl_plot_data_dict]
    np_group_size = np.arange(int(group_start), int(group_stop) + 1, int(group_int))
    xticks = np.arange(int(group_start)-10, int(group_stop)+10 + 1, int(group_int))
    yticks = np.arange(0.50, 1.06, 0.05)
    loc = 'lower right'
    plot_metrics_for_all_the_algos('bw-accept-ratio-comp', plot_save_dir, lbl_plot_data_dict_ls, 5, np_group_size,
                                   color_dict, marker_dict,
                                   'Number of Groups', 'BW. Demands Acceptance Ratio',
                                   'BW. Demands Acceptance Ratio Vs Number of groups', xticks_ls=xticks,
                                   ytick_ls=yticks)



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

    simulations_dirs = ''
    if args.groups_parent_dirs is not None and args.groups_start is not None \
            and args.groups_stop is not None and args.groups_size_int is not None \
            and args.labels is not None:
        labels = args.labels
        groups_parent_dirs = args.groups_parent_dirs
        start = args.groups_start
        stop = args.groups_stop
        inter = args.groups_size_int
        run_list = range(1, 21, 1)

    else:

        # simulations_dirs = utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/dst,' \
        #                    +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/dst-lb,' \
        #                    +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/l2bm-10,' \
        #                    +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/l2bm-40,' \
        #                    +utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/l2bm-60'
        # groups_parent_dirs_grid = '/home/hsoni/qos-multicast-compile/exp3-vabw-25-75/dst,' \
        #                           +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-75/lb,' \
        #                           +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-10-10,' \
        #                           +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-40-10,' \
        #                           +utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-60-10'
        wo_churn_congested_sim_dirs = utils.working_dir+'/wo-churn-congested/dst,' \
                                      +utils.working_dir+'/wo-churn-congested/dst-lb,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-10,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-20,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-30,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-40,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-50,' \
                                      +utils.working_dir+'/wo-churn-congested/l2bm-60'

        churn_congested_sim_dirs = utils.working_dir+'/churn-congested/dst,' \
                                   +utils.working_dir+'/churn-congested/dst-lb,' \
                                   +utils.working_dir+'/churn-congested/l2bm-10,' \
                                   +utils.working_dir+'/churn-congested/l2bm-20,' \
                                   +utils.working_dir+'/churn-congested/l2bm-30,' \
                                   +utils.working_dir+'/churn-congested/l2bm-40,' \
                                   +utils.working_dir+'/churn-congested/l2bm-50,' \
                                   +utils.working_dir+'/churn-congested/l2bm-60'


        labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.2,L2BM-0.3,L2BM-0.4,L2BM-0.5,L2BM-0.6'
        # labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.2,L2BM-0.4,L2BM-0.6'
        # plot_save_dir = '/home/hsoni/qos-multicast-compile/sim-exp3-vabw-30-70/plots/march-3-2017/'
        # util.mkdir(plot_save_dir)
        # start = 30
        # stop = 70
        # inter = 10
        # run_list = range(1, 21, 1)

    # plot_superimposed_bw_accept_ratio_for_different_groups(simulations_dirs, groups_parent_dirs_grid, plot_save_dir,
    #                                                        labels, start, stop, inter, run_list, color_dict, marker_dict)
    plot_save_dir = utils.working_dir+'/wo-churn-congested/plots/'
    plot_save_dir_c = utils.working_dir+'/churn-congested/plots-all-thetas/'
                      # 'churn-congested-10-100/plots-fig-8-rw-2-2017/'
    # mkdir(plot_save_dir)
    util.mkdir(plot_save_dir_c)
    osp_wo_c = OtherStatsProcessorWOChurn()
    osp_w_c = OtherStatsProcessorWChurn()
    start = 10
    stop = 150
    inter = 10
    run_list = range(500)
    color_dict = util.get_color_dict(wo_churn_congested_sim_dirs, labels)
    marker_dict = util.get_marker_dict(wo_churn_congested_sim_dirs, labels)

    # osp_wo_c.plot_other_metrics_for_different_groups(wo_churn_congested_sim_dirs, plot_save_dir, labels, start, stop,
    #                                                  inter, run_list, color_dict, marker_dict)
    ### Plot Churn data
    color_dict = util.get_color_dict(churn_congested_sim_dirs, labels)
    marker_dict = util.get_marker_dict(churn_congested_sim_dirs, labels)
    osp_w_c.plot_other_metrics_for_different_groups(churn_congested_sim_dirs, plot_save_dir_c, labels,start, stop,
                                                    inter, run_list, color_dict, marker_dict)
    group = 130
    runs = range(500)
    churn_congested_sim_dirs_all = \
        utils.working_dir+'/churn-congested/dst,' \
        +utils.working_dir+'/churn-congested/dst-lb,' \
        +utils.working_dir+'/churn-congested/l2bm-10,' \
        +utils.working_dir+'/churn-congested/l2bm-20,' \
        +utils.working_dir+'/churn-congested/l2bm-30,' \
        +utils.working_dir+'/churn-congested/l2bm-40,' \
        +utils.working_dir+'/churn-congested/l2bm-50,' \
        +utils.working_dir+'/churn-congested/l2bm-60'
    labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.2,L2BM-0.3,L2BM-0.4,L2BM-0.5,L2BM-0.6'
    # labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.2,L2BM-0.4,L2BM-0.6'
    color_dict = util.get_color_dict(churn_congested_sim_dirs_all, labels)
    # osp_wo_c.plot_bw_acceptance_bar_graph(wo_churn_congested_sim_dirs, plot_save_dir, labels, group, runs, color_dict)
    osp_w_c.plot_bw_acceptance_bar_graph(churn_congested_sim_dirs_all, plot_save_dir_c, labels, group, runs, color_dict)
