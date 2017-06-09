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


import json
import time
import re
import sys
import os
import io
import signal
import argparse
import scipy.stats as ss
import threading
# import networkx as nx
import subprocess, shlex
import select
import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors

from os import system as cmd
from scipy.stats import kurtosis
from gi.overrides.keysyms import diaeresis


def main():
    inter_arrival_rate = 60
    print (np.random.exponential(inter_arrival_rate, 10))
    start_interval = []
    try:
        with open("/user/hsoni/home/time-split.txt") as f:
            lines = [x.strip('\n') for x in f.readlines()]
        ts_p = 0
        for line in lines:
            time = line.split(' ')
            t = time[0].strip()
            ts = int(t.split(':')[2].strip())
            start_interval.append(ts-ts_p)
            ts_p = ts
        start_interval[0] = 0
        json.dump(start_interval, open("group-start-sched.txt",'w'))
    except Exception as e:
        print str(e)
        print ('error in reading file ')
        sys.exit(1)


def calculate_stddev_mean_sum_max_min_utilizedlink_for_ts(time_edge_con, time_st):
    timestamp_stddev_mean_sum_max_min_nouselink = []
    time_seq = 0
    count  = 0
    # print(len(time_edge_con))
    for [ts, link_cons] in time_edge_con:
        # time.strptime(ts, "%H:%M:%S:+000")
        if ts > time_st:
            count += 1
            if count < 2:
                continue
            link_utils = []
            for cons in link_cons:
                link_util = (float(cons)/(1024*100)) # link consumption in Mbps
                # link_util = float(cons)
                # print(link_util)
                link_utils.append(link_util)
            np_link_utils = np.array(link_utils)
            # print (np_link_utils)
            if len(link_utils) == 0:
                continue
                # timestamp_std_mean_nouselink.append([time_seq, 0,0,0,0,0,0])
            else:
                # print link_utils
                time_seq = time_seq + 1
                timestamp_stddev_mean_sum_max_min_nouselink.append([int(time_seq),
                                                     np.std(np_link_utils),
                                                     np.mean(np_link_utils),
                                                     np.sum(np_link_utils),
                                                     np.amax(np_link_utils),
                                                     np.amin(np_link_utils),
                                                     # kurtosis(np_link_utils, fisher=False, bias=False),
                                                     len(link_utils)])
                break
    return timestamp_stddev_mean_sum_max_min_nouselink


def get_link_stats_after_ts(time_edge_con, time_st):
    time_seq = 0
    count = 0
    # link_utils_list = []
    # print(len(time_edge_con))
    for [ts, link_cons] in time_edge_con:
        # time.strptime(ts, "%H:%M:%S:+000")
        if ts >= time_st:
            count += 1
            if count < 2:
                continue
            link_utils = []
            # np_prev_link_utils = np_curr_link_utils
            # for link, cons in link_cons.iteritems():
            for cons in link_cons:
                link_util = (float(cons)/(1000)) # link consumption in Mbps
                # link_util = int(cons)
                # print(link_util)
                link_utils.append(link_util)
            # np_curr_link_utils = np.array(link_utils)
            # print (np_link_utils)
            # print link_utils
            time_seq = time_seq + 1
            # link_utils_list.append(link_utils)
            # print link_utils
            return link_utils
    # for l in link_utils_list:
    #     print l
    # np_link_utils_list = np.array(link_utils_list)
    # np_diff = np.diff(np_link_utils_list, axis=0)
    # print ('---  Difference ---')
    # print np_diff


def calculate_stddev_mean_sum_max_min_utilizedlink(time_edge_con):
    timestamp_stddev_mean_sum_max_min_nouselink = []
    time_seq = 0
    skip = 0
    # print(len(time_edge_con))
    for [ts, link_cons] in time_edge_con:
        # time.strptime(ts, "%H:%M:%S:+000")
        link_utils = []
        # for link, cons in link_cons.iteritems():
        for cons in link_cons:
            if cons <= 0.0:
                continue
            link_util = (float(cons)/1000) # link consumption in Mbps
            # link_util = float(cons)
            # print(link_util)
            link_utils.append(link_util)
        np_link_utils = np.array(link_utils)
        # print (np_link_utils)
        if len(link_utils) == 0:
            continue
            # timestamp_std_mean_nouselink.append([time_seq, 0,0,0,0,0,0])
        else:
            # print link_utils
            time_seq = time_seq + 1
            timestamp_stddev_mean_sum_max_min_nouselink.append([int(time_seq),
                                                 np.std(np_link_utils),
                                                 np.mean(np_link_utils),
                                                 np.sum(np_link_utils),
                                                 np.amax(np_link_utils),
                                                 np.amin(np_link_utils),
                                                 len(link_utils)])
    return timestamp_stddev_mean_sum_max_min_nouselink



def get_stddev_sumlinkutil_t_data_multiuns(links_stat_dir, no_of_runs_dir):
    file_name_prefix = links_stat_dir+"/run"
    multicast_run_metrics = []
    for i in no_of_runs_dir:
        file_name = file_name_prefix  + str(i)+"/link-bw-stats.txt"
        reciver_log_f = file_name_prefix  + str(i)+"/receivers_count.log"
        # print(file_name)
        # print(reciver_log_f)
        ts_max_rec = get_all_receiver_timestamp(reciver_log_f)
        time_edge_consumption = get_group_bw_dict(file_name)
        timestamp_std_mean_nouselink = calculate_stddev_mean_sum_max_min_utilizedlink_for_ts(time_edge_consumption, ts_max_rec)
        # print file_name_prefix
        # print timestamp_std_mean_nouselink
        multicast_run_metrics += timestamp_std_mean_nouselink
    #print (multicast_run_metrics)
    np_multicast_run_metrics = np.array(multicast_run_metrics)
    std_dev_dist_data = [np.mean(np_multicast_run_metrics[:,1]),
                         np_multicast_run_metrics.shape[0],
                         # ss.sem(np_multicast_run_metrics[:,1])]
                         np.std(np_multicast_run_metrics[:,1])]
    avg_linkutil_dist_data = [np.mean(np_multicast_run_metrics[:,2]),
                         np_multicast_run_metrics.shape[0],
                         # ss.sem(np_multicast_run_metrics[:,2])]
                         np.std(np_multicast_run_metrics[:,2])]
    max_linkutil_dist_data = [np.mean(np_multicast_run_metrics[:,4]),
                         np_multicast_run_metrics.shape[0],
                         # ss.sem(np_multicast_run_metrics[:,4])]
                         np.std(np_multicast_run_metrics[:,4])]
    return [std_dev_dist_data, avg_linkutil_dist_data, max_linkutil_dist_data]


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
            # print (str(ts) + " " + line[0] + " " + line[1])
            # tss.append(ts)
    except Exception as ex:
        print ('error in reading file named ' + receiver_log)
        print (ex)
        sys.exit(1)
    # np_ts = np.array(tss)
    # print np.amax(np_ts)
    # exit()
    return ts_receivers


def get_all_receiver_timestamp(receiver_log):
    ts_rec =  read_receiver_count_time_stamp_file(receiver_log)
    [t, c] = ts_rec[-1]
    return t


def get_stats_for_single_run(links_stat_dir, run_number):
    file_name_prefix = links_stat_dir+"/run"
    multicast_run_metrics = []
    file_name = file_name_prefix  + str(run_number)+"/link-bw-stats.txt"
    print (file_name)
    time_edge_consumption = get_group_bw_dict(file_name)
    return calculate_stddev_mean_sum_max_min_utilizedlink(time_edge_consumption)


def plot_datapoints_for_different_groups(links_stat_dirs, labels, group_start, group_stop, group_int, group_runs):
    experiment_comp_paths = links_stat_dirs.split(',')
    plot_save_dir = '/home/hsoni/qos-multicast-compile/exp2-2mb-10-100/correction-plot_normalized_data-29-nov/temp/'
    # plot_save_dir = '/user/hsoni/home/qos-multicast-compile/exp3-vabw-25-75/plot_normalized_data-23-oct/'
    ls = labels.split(',')
    NUM_COLORS = len(ls)
    lbl_plot_data_dict = {}
    name = None
    cm = plt.get_cmap('CMRmap')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    color_dict = {}
    color_list = [scalarMap.to_rgba(i) for i in range(NUM_COLORS)]
    for l, cl in itertools.izip(ls, color_list):
        color_dict[l] = cl
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    np_group_size = np.arange(int(group_start), int(group_stop)+1, int(group_int))
    for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
        stddev_plot_data = []
        avg_util_plot_data = []
        max_util_plot_data = []
        for gs in np.nditer(np_group_size):
            dir  = links_stat_dir + "/"+str(int(gs))
            [stddev_d, avg_linkutil_d, max_linkutil_d] = get_stddev_sumlinkutil_t_data_multiuns(dir, group_runs)
            stddev_plot_data.append(stddev_d)
            avg_util_plot_data.append(avg_linkutil_d)
            max_util_plot_data.append(max_linkutil_d)
        np_stddev_plot_data = np.array(stddev_plot_data)
        np_avg_util_plot_data = np.array(avg_util_plot_data)
        np_max_util_plot_data = np.array(max_util_plot_data)
        lbl_plot_data_dict[lbl] = [np_stddev_plot_data , np_avg_util_plot_data, np_max_util_plot_data]
        # print lbl
    fig = plt.figure()
    ax = fig.add_subplot(111)
    marker = itertools.cycle(('o', 'v', 's', '8', 'p', 'D'))
    for l, plot_data in lbl_plot_data_dict.items():
        # print l
        max_util = plot_data[2]
        print 'max_util[:,1]', max_util[:,1]
        d = max_util[:,2]
        err = ss.t._ppf((1+0.95)/2, max_util[:,1])
        plt.errorbar(np_group_size, max_util[:,0], yerr=d*err, color=color_dict[l], label=l)
        # plt.plot(np_group_size, max_util[:,0], color=color_dict[l] ,marker=marker.next(), label=l)
        plt.legend(loc='lower right', frameon=False)
        plt.xlabel('Number of Groups')
        plt.ylabel('Max Link utilization')
        # plt.title('Maximum link utilization Vs Number of groups - '+'Run '+str(group_runs[0]))
        plt.title('Maximum link utilization Vs Number of groups')
        name = 'u-max-link-util'
        # name = 'max-link-utils'+'-r'+str(group_runs[0])
        # plt.xlim(20,75)
        plt.xlim(40,110)
        # plt.ylim(0.4, 1.0)
        plt.yticks(np.arange(0.4, 1.05, 0.05))
    plt.savefig(plot_save_dir+name+".jpeg")
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    marker = itertools.cycle(('o', 'v', 's', '8', 'p', 'D'))
    for l, plot_data in lbl_plot_data_dict.items():
        avg_util = plot_data[1]
        d = avg_util[:,2]
        err = ss.t._ppf((1+0.95)/2, avg_util[:,1])
        plt.errorbar(np_group_size, avg_util[:,0], yerr=d*err, color=color_dict[l], label=l)
        # plt.plot(np_group_size, avg_util[:,0], color=color_dict[l], marker=marker.next(), label= l)
        # plt.errorbar(np_group_size, avg_util[:,0],
        #              yerr=ss.t.ppf(0.90, avg_util[:,1])*avg_util[:,2], label=l)
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel('Number of Groups')
        plt.ylabel('Average Link utilization')
        # plt.title('Average link utilization Vs Number of groups - '+'Run '+str(group_runs[0]))
        # name = "avg-link-util"+'-r'+str(group_runs[0])
        plt.title('Average link utilization Vs Number of groups')
        name = "u-avg-link-util"
        # plt.xlim(20,75)
        plt.xlim(40,110)
        # plt.ylim(0.15, 0.55)
        plt.yticks(np.arange(0.15, 0.6, 0.05))
        # plt.yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.savefig(plot_save_dir+name+".jpeg")
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.set_color_cycle([scalarMap.to_rgba(i) for i in range(NUM_COLORS)])
    marker = itertools.cycle(('o', 'v', 's', '8', 'p', 'D'))
    for l, plot_data in lbl_plot_data_dict.items():
        stddev_util = plot_data[0]
        d = stddev_util[:,2]
        err = ss.t._ppf((1+0.95)/2, stddev_util[:,1])
        plt.errorbar(np_group_size, stddev_util[:,0], yerr=d*err, color=color_dict[l], label=l)
        # plt.plot(np_group_size, stddev_util[:,0], color=color_dict[l], marker=marker.next(), label= l)
        # plt.errorbar(np_group_size, stddev_util[:,0],
        #              yerr=ss.t.ppf(0.90, stddev_util[:,1])*stddev_util[:,2], label=l)
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel('Number of Groups')
        plt.ylabel('Standard Deviation of Link utilization')
        plt.title('Stddev. link utilization Vs Number of groups')
        name = "u-stddev-link-util"
        # plt.xlim(20,75)
        plt.xlim(40,110)
        # plt.ylim(0.0,0.3)
        plt.yticks(np.arange(0.00, 0.6, 0.05))
        # plt.yticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    plt.savefig(plot_save_dir+name+".jpeg")
    return  color_dict



    # plt.errorbar(np_group_size, np_stddev_plot_data[:,0],
    #     yerr=ss.t.ppf(0.95, np_stddev_plot_data[:,1])*np_stddev_plot_data[:,2], label=lbl)

    # plt.errorbar(np_group_size, np_avg_util_plot_data[:,0],
    #              yerr=ss.t.ppf(0.95, np_avg_util_plot_data[:,1])*np_avg_util_plot_data[:,2], label= lbl )


def plot_average_nomalized_datapoints_for_different_groups(links_stat_dirs, labels, color_dict, group_start, group_stop, group_int, group_runs):
    lbl_reference_model = 'DST-PL'
    experiment_comp_paths = links_stat_dirs.split(',')
    plot_save_dir = '/home/hsoni/qos-multicast-compile/exp2-2mb-10-100/correction-plot_normalized_data-29-nov/temp/'
    # plot_save_dir = '/user/hsoni/home/qos-multicast-compile/exp3-vabw-25-75/plot_normalized_data-23-oct/'
    ls = labels.split(',')
    lbl_plot_data_dict = {}
    name = None
    np_group_size = np.arange(int(group_start), int(group_stop)+1, int(group_int))
    np_reference_model_dict = {}
    np_std_dev_plot_data = {}
    np_avg_plot_data = {}
    np_max_plot_data = {}
    for lbl in ls:
        if lbl == lbl_reference_model:
            continue
        lbl_plot_data_dict[lbl] = []
        np_std_dev_plot_data[lbl] = np.zeros(np_group_size.size)
        np_avg_plot_data[lbl] = np.zeros(np_group_size.size)
        np_max_plot_data[lbl] = np.zeros(np_group_size.size)
    for run in group_runs:
        for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
            if lbl_reference_model == lbl:
                reference_model_stddev_plot_data = []
                reference_model_avg_util_plot_data = []
                reference_model_max_util_plot_data = []
                for gs in np.nditer(np_group_size):
                    dir  = links_stat_dir + "/"+str(int(gs))
                    [stddev_d, avg_linkutil_d, max_linkutil_d] = get_stddev_sumlinkutil_t_data_multiuns(dir, [run])
                    reference_model_stddev_plot_data.append(stddev_d[0])
                    reference_model_avg_util_plot_data.append(avg_linkutil_d[0])
                    reference_model_max_util_plot_data.append(max_linkutil_d[0])
                np_reference_model_stddev_plot_data = np.array(reference_model_stddev_plot_data)
                np_reference_model_avg_util_plot_data = np.array(reference_model_avg_util_plot_data)
                np_reference_model_max_util_plot_data = np.array(reference_model_max_util_plot_data)
                np_reference_model_dict[run] = [np_reference_model_stddev_plot_data, np_reference_model_avg_util_plot_data,
                                                np_reference_model_max_util_plot_data]
                break
    for run in group_runs:
        [np_reference_model_stddev, np_reference_model_avg_util, np_reference_model_max_util] = np_reference_model_dict[run]
        for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
            if lbl_reference_model == lbl:
                continue
            stddev_plot_data = []
            avg_util_plot_data = []
            max_util_plot_data = []
            for gs in np.nditer(np_group_size):
                dir  = links_stat_dir + "/"+str(int(gs))
                [stddev_d, avg_linkutil_d, max_linkutil_d] = get_stddev_sumlinkutil_t_data_multiuns(dir, [run])
                stddev_plot_data.append(stddev_d[0])
                avg_util_plot_data.append(avg_linkutil_d[0])
                max_util_plot_data.append(max_linkutil_d[0])
            np_stddev_plot_data = np.array(stddev_plot_data)
            np_avg_util_plot_data = np.array(avg_util_plot_data)
            np_max_util_plot_data = np.array(max_util_plot_data)
            nzd_np_stddev = np.divide(np.subtract(np_stddev_plot_data, np_reference_model_stddev),np_reference_model_stddev)
            nzd_np_avg = np.true_divide(np.subtract(np_avg_util_plot_data, np_reference_model_avg_util),np_reference_model_avg_util)
            nzd_np_max = np.true_divide(np.subtract(np_max_util_plot_data, np_reference_model_max_util),np_reference_model_max_util)
            # print str(run) + ' '+ str(lbl)+ ' ' +str(nzd_np_stddev)
            # print nzd_np_avg
            # print nzd_np_max
            lbl_plot_data_dict[lbl].append([nzd_np_stddev, nzd_np_avg, nzd_np_max])
    for l, label_data in lbl_plot_data_dict.items():
        for nz_np_stddev, nz_np_avg, nz_np_maz in label_data:
            # print l
            # print nz_np_stddev
            np_std_dev_plot_data[l] = np.vstack((np_std_dev_plot_data[l], nz_np_stddev))
            np_avg_plot_data[l] = np.vstack((np_avg_plot_data[l], nz_np_avg))
            np_max_plot_data[l] = np.vstack((np_max_plot_data[l], nz_np_maz))
        np_std_dev_plot_data[l] = np.delete(np_std_dev_plot_data[l], (0), axis=0)
        np_avg_plot_data[l] = np.delete(np_avg_plot_data[l], (0), axis=0)
        np_max_plot_data[l] = np.delete(np_max_plot_data[l], (0), axis=0)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lbl, avg_util in np_avg_plot_data.items():
        if lbl == lbl_reference_model:
            continue
        # plt.plot(np_group_size, avg_util, marker=marker.next(), label= lbl+'-Vs-'+lbl_reference_model)

        d = ss.sem(avg_util, axis=0)
        # print 'avg_util.shape - ', avg_util.shape
        # print 'norm avg_util[:,1].shape[0] - ', avg_util[:,1].shape[0]
        # print 'stddev axis 1 -', d
        err = ss.t._ppf((1+0.95)/2, np.subtract(avg_util[:,1].shape[0], 1))
        # print err*d
        plt.errorbar(np_group_size, np.mean(avg_util, axis=0), yerr=err * d, color=color_dict[lbl],
                     label=lbl+'-Vs-'+lbl_reference_model)
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel('Number of Groups')
        plt.ylabel('Normalized Average Link utilization')
        # plt.title('Average link utilization Vs Number of groups - '+'Run '+str(group_runs[0]))
        # name = "avg-link-util"+'-r'+str(group_runs[0])
        plt.title('Normalized Average link utilization Vs Number of groups')
        name = "u-normalized-avg-link-util"+'-ci'
        # plt.xlim(20,75)
        plt.xlim(40,110)
        plt.ylim(-0.10, 0.25)
    plt.savefig(plot_save_dir+name+".jpeg")
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lbl, max_util in np_max_plot_data.items():
        if lbl == lbl_reference_model:
            continue
        # plt.plot(np_group_size, max_util, marker=marker.next(), label= lbl+'-Vs-'+lbl_reference_model)
        d = ss.sem(max_util, axis=0)
        err = ss.t._ppf((1+0.95)/2, np.subtract(max_util[:,1].shape[0], 1))
        plt.errorbar(np_group_size, np.mean(max_util, axis=0), yerr=err*d, color=color_dict[lbl],
                     label=lbl+'-Vs-'+lbl_reference_model)
        plt.legend(loc='upper left', frameon=False)
        plt.xlabel('Number of Groups')
        plt.ylabel('Normalized MaxLink utilization')
        # plt.title('Average link utilization Vs Number of groups - '+'Run '+str(group_runs[0]))
        # name = "avg-link-util"+'-r'+str(group_runs[0])
        plt.title('Normalized Max link utilization Vs Number of groups')
        name = "u-normalized-max-link-util"+'-ci'
        # plt.xlim(20,75)
        plt.xlim(40,110)
        # plt.ylim(-0.5, 0.1)
        plt.yticks(np.arange(-0.5, 0.1, 0.05))
    plt.savefig(plot_save_dir+name+".jpeg")
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for lbl, stddev_util in np_std_dev_plot_data.items():
        if lbl == lbl_reference_model:
            continue
        # plt.plot(np_group_size, stddev_util, marker=marker.next(), label= lbl+'-Vs-'+lbl_reference_model)
        d = ss.sem(stddev_util, axis=0)
        err = ss.t._ppf((1+0.95)/2, np.subtract(stddev_util[:,1].shape[0], 1))
        plt.errorbar(np_group_size, np.mean(stddev_util, axis=0), yerr=err*d, color=color_dict[lbl],
                     label=lbl+'-Vs-'+lbl_reference_model)
        plt.legend(loc='upper right', frameon=False)
        plt.xlabel('Number of Groups')
        plt.ylabel('Normalized Stddev utilization')
        # plt.title('Average link utilization Vs Number of groups - '+'Run '+str(group_runs[0]))
        # name = "avg-link-util"+'-r'+str(group_runs[0])
        plt.title('Normalized Stddev link utilization Vs Number of groups')
        name = "u-normalized-stddev-link-util"+'-ci'
        # plt.xlim(20,75)
        plt.xlim(40,110)
        # plt.ylim(-0.5, 0.0)
        plt.yticks(np.arange(-0.5, 0.0, 0.05))
    plt.savefig(plot_save_dir+name+".jpeg")
    plt.clf()



def plot_datapoints_for_one_group_run_time_scale(links_stat_dirs, labels, group_start, group_stop, group_int):
    experiment_comp_paths = links_stat_dirs.split(',')
    ls = labels.split(',')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
        np_group_size = np.arange(int(group_start), int(group_stop)+1, int(group_int))
        for gs in np.nditer(np_group_size):
            dir = links_stat_dir + "/"+str(int(gs))
            run_data = get_stats_for_single_run(dir, 1)
            data = np.array(run_data)
            ax.plot(data[:,0] , data [:,3], label=lbl+' Total Consumption')
            # line, = ax.plot(data[:,0] , data [:,1], label='StdDev' )
            # line, = ax.plot(data[1:100,1] , data [1:100,3], label='Totoal Bandwidth Consumption Vs StdDev' )
            # ax.set_xlabel('StdDev in Mbps')
            ax.set_xlabel('Time intervals')
            ax.set_ylabel('Total bandwidth consumption in Mbps')
            plt.legend(loc='upper left', frameon=False)

    plt.show()



def plot_link_util_cdf_for_different_groups(links_stat_dirs, labels, group_start, group_stop, group_int, group_runs_l):
    experiment_comp_paths = links_stat_dirs.split(',')
    ls = labels.split(',')
    np_group_size = np.arange(int(group_start), int(group_stop)+1, int(group_int))
    for gs in np.nditer(np_group_size):
        for rn in group_runs_l:
            name = ''
            title = ''
            for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
                dir  = links_stat_dir + "/"+str(int(gs))+"/"
                title = ''
                title = "Number Of Groups : " + str(gs) + " Run " + str(rn)
                name = str(gs)+"g_run"+ str(rn)
                [bin_edges, cdf] = get_maxutil_multiuns(dir, [rn])
                plt.plot(bin_edges, cdf, label='Link Utilization - CDF-'+lbl)
            plt.legend(loc='lower right', frameon=False)
            plt.title(title)
            plt.xlim(0, 100)
            # plt.ylim(0, 50)
            # plt.savefig('/user/hsoni/home/qos-multicast-compile/low_traffic_with_fix/'+title.replace(' ','_')+"empirical.jpeg")
            plt.xlabel('% link utilization')
            plt.ylabel('CDF link utilization')
            plt.savefig('/user/hsoni/home/qos-multicast-compile/exp3-vabw-25-75/plots2/r1/'+name+".jpeg")
            plt.show()



##CONT
def get_maxutil_multiuns(links_stat_dir, group_runs_l):
    file_name_prefix = links_stat_dir+"run"
    link_utlis_metric = []
    for i in group_runs_l:
        file_name = file_name_prefix  + str(i) + "/link-bw-stats.txt"
        reciver_log_f = file_name_prefix  + str(i)+"/receivers_count.log"
        # print file_name
        ts_max_rec = get_all_receiver_timestamp(reciver_log_f)
        time_edge_consumption = get_group_bw_dict(file_name)
        link_utils = get_link_stats_after_ts(time_edge_consumption, ts_max_rec)
        link_utlis_metric += link_utils
    np_link_utlis_metric = np.array(link_utlis_metric)
    num_bins = 11
    counts, bin_edges = np.histogram(np_link_utlis_metric, range=(0.0, np.amax(np_link_utlis_metric)) , bins=num_bins, normed=False)
    cdf = np.cumsum(counts, dtype=float)/102
    # print(cdf)
    # print bin_edges[1:]
    return [bin_edges[1:], cdf]
    # sorted=np.sort(np_link_utlis_metric)
    # print sorted
    # yvals=np.arange(len(sorted))/float(len(sorted))
    # print yvals
    # return [sorted, yvals]


# returns list of dictonary
# [time , {<link -> bw>}]
def get_group_bw_dict(link_stat_f):
    time_edge_consumption = []
    try:
        with open(link_stat_f) as f:
            lines = [x.strip('\n') for x in f.readlines()]
        for l in lines:
            line = l.strip(',').strip('(')
            time_stat_l = line.split('-----')
            if len(time_stat_l) < 2 or time_stat_l[1]  == '' :
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
            #     if bw_cons != 0:
            #         used_edges = True
            # if used_edges:
            time_edge_consumption.append([ts, edge_consumption])
            # print edge_consumption
        # print (len(time_edge_consumption))
        # print (time_edge_consumption['02:20:35:837'])
    except Exception as ex:
        print ('error in reading file named ' + link_stat_f)
        print (ex)
        sys.exit(1)
    return time_edge_consumption





#python ~/multicast-qos-python-may-2016/link-stats-processor.py
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
            and args.groups_stop is not  None and args.groups_size_int is not None\
            and args.labels is not None:
        # plot_link_util_cdf_for_different_groups(args.groups_parent_dirs, args.labels, args.groups_start,
        #                                      args.groups_stop, args.groups_size_int, [1, 2, 3, 4, 5])
        color_dict = plot_datapoints_for_different_groups(args.groups_parent_dirs, args.labels, args.groups_start,
                                             args.groups_stop, args.groups_size_int, range(1,21,1))
        plot_average_nomalized_datapoints_for_different_groups(args.groups_parent_dirs, args.labels, color_dict, args.groups_start,
                                             args.groups_stop, args.groups_size_int, range(1,21,1))
        # plot_datapoints_for_one_group_run_time_scale(args.groups_parent_dirs, args.labels, args.groups_start,
        #                                      args.groups_stop, args.groups_size_int)
    else:
        main()


# i = 0
# for bw in np.random.randint(1, 5, 50):
#     i += 1
#     print ("239.0.0."+ str(i)+ " - "+str(bw*1000))
# return