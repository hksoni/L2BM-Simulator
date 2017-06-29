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

#!/usr/bin/python

import argparse
import json
import matplotlib
matplotlib.use('Agg')

import scipy.stats as ss
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
import matplotlib.pyplot as plt
plt.style.use('classic')
from Utils import *

def main():
    # get_other_stats_wo_churn('/home/hsoni/qos-multicast-compile/simulation/static/const-bw/dst/10/run1/other-results.txt')
    # get_simulation_data_interval('/user/hsoni/home/qos-multicast-compile/simulation/churn/va-bw/dst/50/', 0)
    # a = np.zeros(3)
    # data1 = np.array([10, 0, 90])
    # data2 = np.array([5, 2, 3])
    # print data1 * data2
    # exit()
    # err = ss.t._ppf((1+0.95)/2, np.subtract(data1, 1))
    # print err
    # a = []
    # a.append([3,4,7])
    # a.append([3,40,7])
    # b = [[3,7],[10, 0],[11, 7],[30,0],[60, 50],[10, 10]]
    # n_a = np.array(a)
    # n_b = np.array(b)
    # print n_b
    # x = np.where(n_b == 0)
    # print x
    # print x[1].shape
    # if x[0].shape[0] != 0 and x[1].shape[0] != 0:
    #     print x[0][0]
    # print np.delete(n_b, x[0], 0)
    # print n_b[:,0,1]
    # print n_b
    # a = np.vstack((n_b, a))
    # print n_a[:,0]
    # print (np.true_divide(n_a, n_b))
    c = 2
    np_arr = np.array([[1,2],[1,1],[3,5],[2,3]])
    s = np_arr.shape
    k = s[0]-np.mod(s[0], c)
    print  np_arr
    print s[0]
    x = np_arr[:k].reshape((s[0]/c,c ,s[1]))
    n_d1 = np.mean(x, axis=1)
    # n_d2 = np.mean(np_arr[-(s-k):])
    # n = n_d1.tolist()
    # n.append(n_d2)
    print n_d1


# [float(r_a), np.mean(np_utils), np.std(np_utils), np.max(np_utils),
# self.no_links_gt_lu(np_utils, 0.60), self.no_links_gt_lu(np_utils, 0.70),
# self.no_links_gt_lu(np_utils, 0.80), self.no_links_gt_lu(np_utils, 0.90)]
def get_simulation_data(links_stat_dir, run, trunc_index=None):
    file_name = links_stat_dir+"/run"+ str(run)+"/churn-link-bw-stats.txt"
    # print file_name
    with open(file_name) as f:
        line = [x.strip('\n') for x in f.readlines()][0]
        data = json.loads(line)
        np_data = np.array(data, dtype=float)
        np_data[:,4:8] *= 100
        np_data[:,4:8] /= 102
        if trunc_index is not None:
            n_d = np.delete(np_data , trunc_index, 0)
            return  n_d.tolist()
        else:
            return np_data.tolist()



def get_simulation_data_interval(links_stat_dir, run, no_of_int_collapse=0):
    file_name = links_stat_dir+"/run"+ str(run)+"/churn-link-bw-stats.txt"
    with open(file_name) as f:
        line = [x.strip('\n') for x in f.readlines()][0]
        data = json.loads(line)
        np_arr = np.array(data, dtype=float)
        if no_of_int_collapse > 1:
            s = np_arr.shape
            print "shape[0] ", s[0]
            k = s[0]-np.mod(s[0], no_of_int_collapse)
            if s[0] == k:
                n_d_mean_std = np.mean(np_arr.reshape((s[0]/no_of_int_collapse, no_of_int_collapse, s[1])), axis=1)
                n_d_max = np.amax(np_arr.reshape((s[0]/no_of_int_collapse, no_of_int_collapse, s[1])), axis=1)
                # print n_d.shape
                return n_d_mean_std, n_d_max
            else:
                n_d_mean_std1 = np.mean(np_arr[:k].reshape((s[0]/no_of_int_collapse, no_of_int_collapse, s[1])), axis=1)
                n_d_mean_std2 = np.mean(np_arr[-(s[0]-k):].reshape((1, s[0]-k, s[1])), axis=1)
                n_d_max1 = np.amax(np_arr[:k].reshape((s[0]/no_of_int_collapse, no_of_int_collapse, s[1])), axis=1)
                n_d_max2 = np.amax(np_arr[-(s[0]-k):].reshape((1, s[0]-k, s[1])), axis=1)
                n_d_mean_std = np.vstack((n_d_mean_std1, n_d_mean_std2))
                n_d_max = np.vstack((n_d_max1, n_d_max2))
                return n_d_mean_std, n_d_max
        else:
            return np_arr



def get_ref_simulation_data(links_stat_dir, run):
    file_name = links_stat_dir+"/run"+ str(run)+"/churn-link-bw-stats.txt"
    trunc_index = None
    with open(file_name) as f:
        line = [x.strip('\n') for x in f.readlines()][0]
        data = json.loads(line)
        np_data = np.array(data, dtype=float)
        x = np.where(np_data == 0.0)
        if x[0].shape[0] != 0 and x[1].shape[0] != 0:
            # print file_name
            trunc_index = x[0]
            n_d = np.delete(np_data , trunc_index, 0)
            return n_d.tolist(), trunc_index
        else:
            return data, None



def get_other_data(links_stat_dir,  run):
    file_name = links_stat_dir+"/run"+ str(run)+"/churn-other-results.txt"
    receiver_requests = None
    bandwidth = None
    bw_accept_map = None
    try:
        with open(file_name) as f:
            lines = [x.strip('\n') for x in f.readlines()]
        receiver_requests = lines[0].split('=')[1]
        bandwidth = lines[1].split('=')[1]
        if len(lines) > 2:
            bw_accept_map = eval(lines[2].split('=')[1])
    except Exception as ex:
        print ('error in reading file named' + file_name)
        print (ex)
        sys.exit(1)
    return [receiver_requests, bandwidth], bw_accept_map




def plot_metrics_for_all_the_algos(file_name, dest_dir_loca, lbl_plot_data_dict, metrics_index,
                                   np_group_size, color_dict, marker_dict, xlable, ylable, title, x1=None, x2=None,
                                   xticks_ls=np.arange(40, 111, 10),  y1=None, y2=None, ytick_ls=None,
                                   loc=0):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # i = 0
    eb_l = []
    l_s = []
    linestyles = ['-', '--']
    # for lbl_plot_data_dict in lbl_plot_data_dict_ls:
    eb_l = []
    l_s = []
    sorted_order = sorted(lbl_plot_data_dict.keys())
    for l in sorted_order:
        plot_data = lbl_plot_data_dict[l]
        metric = plot_data[metrics_index]
        l_s.append(l)
        err = ss.t._ppf((1+0.95)/2, np.subtract(metric[:,1], 1))
        eb = plt.errorbar(np_group_size, metric[:,0], color=color_dict[l], fmt=marker_dict[l]+'-',
                          yerr=err*metric[:,2], label=l)
        eb_l.append(eb)
        plt.legend(loc=loc, frameon=False)
        plt.xlabel(xlable)
        plt.ylabel(ylable)
        plt.title(title)
        if xticks_ls is not  None:
            plt.xticks(xticks_ls)
        elif x1 is not None and x2 is not  None:
            plt.xlim(x1,x2)
        else:
            ''
        if ytick_ls is not None:
            plt.yticks(ytick_ls)
        elif y1 is not None and y2 is not  None:
            plt.ylim(y1,y2)
        else:
            ''
    legends = ax.legend(eb_l, l_s, loc=loc, ncol=2, frameon=False)
    for t in legends.get_texts():
        if color_dict.has_key(t.get_text()):
            t.set_color(color_dict[t.get_text()])
    # fig.savefig(dest_dir_loca+file_name+".svg", format='svg', dpi=1200)
    fig.savefig(dest_dir_loca+file_name+".eps", format='eps', dpi=1200)



def get_single_run_lu_metrics(links_stat_dir, run_num):
    lu_metrics_data = np.array(get_simulation_data(links_stat_dir, run_num), dtype=float)
    lu_metrics_avg = np.mean(lu_metrics_data, axis=0)
    return lu_metrics_avg.tolist()



# [float(r_a), np.mean(np_utils), np.std(np_utils), np.max(np_utils),
# self.no_links_gt_lu(np_utils, 0.60), self.no_links_gt_lu(np_utils, 0.70),
# self.no_links_gt_lu(np_utils, 0.80), self.no_links_gt_lu(np_utils, 0.90)]
def get_multi_runs_lu_metrics(links_stat_dir, no_of_runs_dir):
    multicast_link_run_metrics = []
    for i in no_of_runs_dir:
        mean_std_max_link_util = get_single_run_lu_metrics(links_stat_dir, i)
        multicast_link_run_metrics.append(mean_std_max_link_util)
    np_multicast_runs_lu_metrics = np.array(multicast_link_run_metrics, dtype=float)
    avg_lu_dist_data = [np.mean(np_multicast_runs_lu_metrics[:, 1]),
                        np_multicast_runs_lu_metrics.shape[0],
                        ss.sem(np_multicast_runs_lu_metrics[:, 1])]
    stddev_lu_dist_data = [np.mean(np_multicast_runs_lu_metrics[:, 2]),
                           np_multicast_runs_lu_metrics.shape[0],
                           ss.sem(np_multicast_runs_lu_metrics[:, 2])]
    max_lu_dist_data = [np.mean(np_multicast_runs_lu_metrics[:, 3]),
                        np_multicast_runs_lu_metrics.shape[0],
                        ss.sem(np_multicast_runs_lu_metrics[:, 3])]
    no_link_gt_60_lu_dist_data = [np.mean(np_multicast_runs_lu_metrics[:, 4]),
                                  np_multicast_runs_lu_metrics.shape[0],
                                  ss.sem(np_multicast_runs_lu_metrics[:, 4])]
    no_link_gt_70_lu_dist_data = [np.mean(np_multicast_runs_lu_metrics[:, 5]),
                                  np_multicast_runs_lu_metrics.shape[0],
                                  ss.sem(np_multicast_runs_lu_metrics[:, 5])]
    no_link_gt_80_lu_dist_data = [np.mean(np_multicast_runs_lu_metrics[:, 6]),
                                  np_multicast_runs_lu_metrics.shape[0],
                                  ss.sem(np_multicast_runs_lu_metrics[:, 6])]
    no_link_gt_90_lu_dist_data = [np.mean(np_multicast_runs_lu_metrics[:, 7]),
                                  np_multicast_runs_lu_metrics.shape[0],
                                  ss.sem(np_multicast_runs_lu_metrics[:, 7])]
    return avg_lu_dist_data, stddev_lu_dist_data, max_lu_dist_data, no_link_gt_60_lu_dist_data, \
           no_link_gt_70_lu_dist_data, no_link_gt_80_lu_dist_data, no_link_gt_90_lu_dist_data



def link_utils_for_different_groups(links_stat_dirs, plot_save_dir, labels, group_start,
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
        stddev_lu_plot_data = []
        avg_lu_plot_data = []
        max_lu_plot_data = []
        no_link_gt_60_lu_plot_data = []
        no_link_gt_70_lu_plot_data = []
        no_link_gt_80_lu_plot_data = []
        no_link_gt_90_lu_plot_data = []
        for gs in np.nditer(np_group_size):
            dir = links_stat_dir + "/" + str(int(gs))
            avg_lu_dist_data, stddev_lu_dist_data, max_lu_dist_data, no_link_gt_60_lu_dist_data, \
            no_link_gt_70_lu_dist_data, no_link_gt_80_lu_dist_data, no_link_gt_90_lu_dist_data \
                = get_multi_runs_lu_metrics(dir, group_runs)
            avg_lu_plot_data.append(avg_lu_dist_data)
            stddev_lu_plot_data.append(stddev_lu_dist_data)
            max_lu_plot_data.append(max_lu_dist_data)
            no_link_gt_60_lu_plot_data.append(no_link_gt_60_lu_dist_data)
            no_link_gt_70_lu_plot_data.append(no_link_gt_70_lu_dist_data)
            no_link_gt_80_lu_plot_data.append(no_link_gt_80_lu_dist_data)
            no_link_gt_90_lu_plot_data.append(no_link_gt_90_lu_dist_data)
        np_avg_lu_plot_data = np.array(avg_lu_plot_data)
        np_stddev_lu_plot_data = np.array(stddev_lu_plot_data)
        np_max_lu_plot_data = np.array(max_lu_plot_data)
        np_no_link_gt_60_lu_plot_data = np.array(no_link_gt_60_lu_plot_data)
        np_no_link_gt_70_lu_plot_data = np.array(no_link_gt_70_lu_plot_data)
        np_no_link_gt_80_lu_plot_data = np.array(no_link_gt_80_lu_plot_data)
        np_no_link_gt_90_lu_plot_data = np.array(no_link_gt_90_lu_plot_data)
        lbl_plot_data_dict[lbl] = [np_avg_lu_plot_data, np_stddev_lu_plot_data, np_max_lu_plot_data,
                                   np_no_link_gt_60_lu_plot_data, np_no_link_gt_70_lu_plot_data,
                                   np_no_link_gt_80_lu_plot_data, np_no_link_gt_90_lu_plot_data]
    return lbl_plot_data_dict



def plot_link_util_metrics_for_different_groups(links_stat_dirs, plot_save_dir, labels, group_start,
                                                group_stop, group_int, group_runs, color_dict, marker_dict):

    lbl_plot_data_dict = link_utils_for_different_groups(links_stat_dirs, plot_save_dir, labels, group_start,
                                                         group_stop, group_int, group_runs)
    plot_average_data_points(lbl_plot_data_dict, plot_save_dir, group_start, group_stop, group_int,
                             color_dict, marker_dict)


def plot_average_data_points(lbl_plot_data_dict, plot_save_dir, group_start, group_stop, group_int,
                             color_dict, marker_dict):
    np_group_size = np.arange(int(group_start), int(group_stop) + 1, int(group_int))
    xticks = np.arange(int(group_start)-int(group_int), int(group_stop)+int(group_int)+1, int(group_int))
    yticks = np.arange(0.0, 1.06, 0.05)
    loc = 0
    plot_metrics_for_all_the_algos('avg-lu', plot_save_dir, lbl_plot_data_dict, 0, np_group_size,
                                   color_dict, marker_dict, 'Number of Groups', 'Average Link utilization',
                                   'Avg. link utilization Vs Number of groups', xticks_ls=xticks,
                                   ytick_ls=yticks, loc=loc)
    yticks = np.arange(0.0, 1.06, 0.05)
    plot_metrics_for_all_the_algos('stddev-lu', plot_save_dir, lbl_plot_data_dict, 1, np_group_size,
                                   color_dict, marker_dict, 'Number of Groups', 'Standard Deviation of Link utilization',
                                   'Stddev link utilization Vs Number of groups', xticks_ls=xticks,
                                   ytick_ls=yticks, loc=loc)
    plot_metrics_for_all_the_algos('max-lu', plot_save_dir, lbl_plot_data_dict, 2, np_group_size,
                                   color_dict, marker_dict, 'Number of Groups', 'Maximum Link utilization',
                                   'Max link utilization Vs Number of groups', xticks_ls=xticks,
                                   ytick_ls=yticks, loc=loc)
    percent = '60%'
    plot_metrics_for_all_the_algos('num-link-gt-60-lu', plot_save_dir, lbl_plot_data_dict, 3, np_group_size,
                                   color_dict, marker_dict, 'Number of Groups', 'No. of links w/ utilization > '+percent,
                                   'No. of links w/ utilization > '+percent+' Vs Number of groups',
                                   xticks_ls=xticks)
    percent = '70%'
    plot_metrics_for_all_the_algos('num-link-gt-70-lu', plot_save_dir, lbl_plot_data_dict, 4, np_group_size,
                                   color_dict, marker_dict, 'Number of Groups', 'No. of links w/ utilization > '+percent,
                                   'No. of links w/ utilization > '+percent+' Vs Number of groups',
                                   xticks_ls=xticks)
    percent = '80%'
    plot_metrics_for_all_the_algos('num-link-gt-80-lu', plot_save_dir, lbl_plot_data_dict, 5, np_group_size,
                                   color_dict, marker_dict, 'Number of Groups', 'No. of links w/ utilization > '+percent,
                                   'No. of links w/ utilization > '+percent+' Vs Number of groups',
                                   xticks_ls=xticks)
    percent = '90%'
    plot_metrics_for_all_the_algos('num-link-gt-90-lu', plot_save_dir, lbl_plot_data_dict, 6, np_group_size,
                                   color_dict, marker_dict, 'Number of Groups', 'No. of links w/ utilization > '+percent,
                                   'No. of links w/ utilization > '+percent+' Vs Number of groups',
                                   xticks_ls=xticks)



def select_interval_second_samples(ts_d, interval):
    ts = float(0.0)
    last_entry = ts_d[0]
    sampled_data = []
    for entry in ts_d:
        if float(entry[0]) < ts:
            last_entry = entry
        else:
            entry_ts = entry[0]
            while (ts < entry_ts):
                last_entry[0] = ts
                sampled_data.append(last_entry)
                ts += interval
            last_entry = entry
    print len(sampled_data)
    return  sampled_data


def get_exp_moving_avg(ts_samples, smoothing_factor):
    smoothed_data = []
    smoothed_data.append(ts_samples.pop(0))
    f_sf = float(smoothing_factor)
    for item in ts_samples:
        np_smoothed_data_last = np.array(smoothed_data[-1], dtype=float)
        np_current_item = np.array(item, dtype=float)
        current_smoothed_item = (np_current_item * f_sf) + \
                                ((1-f_sf) * np_smoothed_data_last)
        smoothed_data.append(current_smoothed_item.tolist())
    return smoothed_data



def plot_time_sequence_based_metrics(links_stat_dirs, plot_save_dir, labels, group_num, run_num,
                                     interval_collapse, color_dict, marker_dict):
    experiment_comp_paths = links_stat_dirs.split(',')
    ls = labels.split(',')
    algo_path_dict = {}
    for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
        algo_path_dict[lbl]=links_stat_dir

    alg_group_run_metrics_dict = {}

    for lbl, links_stat_dir in algo_path_dict.iteritems():
        dir = links_stat_dir + "/"+str(int(group_num))
        data = get_simulation_data(dir, run_num)
        samples =  get_exp_moving_avg(select_interval_second_samples(data, 1.0), 0.6)
        alg_group_run_metrics_dict[lbl] = np.array(samples, dtype=float)

    xlable = 'Time'
    yticks = np.arange(0.0, 1.15, 0.1)
    file_name = str(group_num)+'run'+str(run_num)+'churn-avg-lu'
    metric_index = 1
    ylable = 'Avg. Link utilization'
    title = 'Average link utilization Vs Time'
    plot_time_series_metric_for_all_the_algos(file_name, plot_save_dir, alg_group_run_metrics_dict, 0, metric_index,
                                              color_dict, marker_dict, xlable, ylable, title, ytick_ls=yticks)
    yticks = np.arange(0.0, 1.15, 0.1)
    file_name = str(group_num)+'run'+str(run_num)+'churn-stddev-lu'
    metric_index = 2
    ylable = 'StdDev. Link utilization'
    title = 'StdDev link utilization Vs Time'
    plot_time_series_metric_for_all_the_algos(file_name, plot_save_dir, alg_group_run_metrics_dict, 0, metric_index,
                                              color_dict, marker_dict, xlable, ylable, title, ytick_ls=yticks)
    yticks = np.arange(0.0, 1.15, 0.1)
    file_name = str(group_num)+'run'+str(run_num)+'churn-max-lu'
    metric_index = 3
    ylable = 'Max. Link utilization'
    title = 'Max link utilization Vs Time'
    plot_time_series_metric_for_all_the_algos(file_name, plot_save_dir, alg_group_run_metrics_dict, 0, metric_index,
                                              color_dict, marker_dict, xlable, ylable, title, ytick_ls=yticks)
    file_name = str(group_num)+'run'+str(run_num)+'churn-no-link-gt-60-lu'
    metric_index = 4
    ylable = '% of links w/ utilization > 0.6'
    title = ylable+ ' Vs '+ xlable
    yticks = np.arange(0, 66, 5)
    # yticks = None
    plot_time_series_metric_for_all_the_algos(file_name, plot_save_dir, alg_group_run_metrics_dict, 0, metric_index,
                                              color_dict, marker_dict, xlable, ylable, title, ytick_ls=yticks)
    file_name = str(group_num)+'run'+str(run_num)+'churn-no-link-gt-70-lu'
    metric_index = 5
    ylable = '% of links w/ utilization > 0.7'
    title = ylable+ ' Vs '+ xlable
    plot_time_series_metric_for_all_the_algos(file_name, plot_save_dir, alg_group_run_metrics_dict, 0, metric_index,
                                              color_dict, marker_dict, marker_dict, xlable, ylable, title, ytick_ls=yticks)
    file_name = str(group_num)+'run'+str(run_num)+'churn-no-link-gt-80-lu'
    metric_index = 6
    ylable = '% of links w/ utilization > 0.8'
    title = ylable+ ' Vs '+ xlable
    plot_time_series_metric_for_all_the_algos(file_name, plot_save_dir, alg_group_run_metrics_dict, 0, metric_index,
                                              color_dict, marker_dict, xlable, ylable, title, ytick_ls=yticks)
    file_name = str(group_num)+'run'+str(run_num)+'churn-no-link-gt-90-lu'
    metric_index = 7
    ylable = '% of links w/ utilization > 0.9'
    title = ylable+ ' Vs '+ xlable
    plot_time_series_metric_for_all_the_algos(file_name, plot_save_dir, alg_group_run_metrics_dict, 0, metric_index,
                                              color_dict, marker_dict, xlable, ylable, title, ytick_ls=yticks)




def plot_time_series_metric_for_all_the_algos(file_name, dest_dir_loca, lbl_plot_data_dict, time_seq_index, metrics_index,
                                               color_dict, marker_dict, xlable, ylable, title, x1=None, x2=None,
                                               xticks_ls=None,  y1=None, y2=None, ytick_ls=np.arange(0,100, 5), loc=0):
    plt.clf()
    fig = plt.figure()
    linestyles = ['-', '', '--', '-.', ':']
    markers = ['', '+', '', '', '']
    i = 0
    sorted_order = sorted(lbl_plot_data_dict.keys())
    for l in sorted_order:
        plot_data = lbl_plot_data_dict[l]
    # for l, plot_data in lbl_plot_data_dict.items():
        width = 2
        ls = linestyles[i]
        if i == 1:
            width = 3
            ls = '-'
        if i == 0:
            width = 2
        plt.plot(plot_data[:,time_seq_index], plot_data[:,metrics_index], color=color_dict[l],
                 linestyle=ls, linewidth=width, label=l)
        i += 1
        plt.xlabel(xlable)
        plt.ylabel(ylable)
        plt.title(title)
        if xticks_ls is not  None:
            plt.xticks(xticks_ls)
        elif x1 is not None and x2 is not  None:
            plt.xlim(x1,x2)
        else:
            ''
        if ytick_ls is not None:
            plt.yticks(ytick_ls)
        elif y1 is not None and y2 is not  None:
            plt.ylim(y1,y2)
        else:
            ''
    legends = plt.legend(loc=loc, ncol=2, frameon=False)
    for t in legends.get_texts():
        if color_dict.has_key(t.get_text()):
            t.set_color(color_dict[t.get_text()])
    fig.savefig(dest_dir_loca+file_name+".eps", format='eps', dpi=1200)
    # fig.savefig(dest_dir_loca+file_name+".svg", format='svg', dpi=1200)



#python ~/multicast-qos-python-may-2016/SimulationStatsProcessor.py
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
    if args.groups_parent_dirs is not None and args.groups_start is not None\
            and args.groups_stop is not None and args.groups_size_int is not None\
            and args.labels is not None:
        groups_parent_dirs = args.groups_parent_dirs
        labels = args.labels
        start = args.groups_start
        stop = args.groups_stop
        inter = args.groups_size_int
    else:
        groups_parent_dirs = working_dir+'/churn-congested/dst,' \
                             +working_dir+'/churn-congested/dst-lb,' \
                             +working_dir+'/churn-congested/l2bm-10,' \
                             +working_dir+'/churn-congested/l2bm-20,' \
                             +working_dir+'/churn-congested/l2bm-30,' \
                             +working_dir+'/churn-congested/l2bm-40,' \
                             +working_dir+'/churn-congested/l2bm-50,' \
                             +working_dir+'/churn-congested/l2bm-60'

        labels = 'DST-PL,DST-LU,L2BM-0.1,L2BM-0.2,L2BM-0.3,L2BM-0.4,L2BM-0.5,L2BM-0.6'
        # labels = 'DST-PL,DST-LU,L2BM-0.1'
        start = 10
        stop = 150
        inter = 10
    # run_list = range(1, 11, 1)
    run_list = range(500)
    save_plots = working_dir+'/churn-congested/plots/'
    color_dict = get_color_dict(groups_parent_dirs, labels)
    marker_dict = get_marker_dict(groups_parent_dirs, labels)
    mkdir(save_plots)
    # plot_time_sequence_based_metrics(groups_parent_dirs, save_plots, labels, 60, 190, 200, color_dict, marker_dict)
    # labels = 'DST-PL,DST-LU,L2BM-0.1'
    groups_parent_dirs = working_dir+'/churn-congested/dst,' \
                         +working_dir+'/churn-congested/dst-lb,' \
                         +working_dir+'/churn-congested/l2bm-10,' \
                         +working_dir+'/churn-congested/l2bm-20,' \
                         +working_dir+'/churn-congested/l2bm-30,' \
                         +working_dir+'/churn-congested/l2bm-40,' \
                         +working_dir+'/churn-congested/l2bm-50,' \
                         +working_dir+'/churn-congested/l2bm-60'
    plot_link_util_metrics_for_different_groups(groups_parent_dirs, save_plots, labels, start, stop, inter, run_list,
                                                color_dict, marker_dict)


