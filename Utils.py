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
import os
import errno
import itertools
import networkx as nx
import numpy as np
import sys

working_dir = os.path.expanduser('~')
# bw_map_file = '/home/hsoni/qos-multicast-compile/exp3-vabw-25-70-LLDMs/ip-bw-qos-mapping-va.txt'
bw_map_file = working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/ip-bw-qos-mapping-va.txt'

def get_color_dict(links_stat_dirs, labels):
    experiment_comp_paths = links_stat_dirs.split(',')
    ls = labels.split(',')
    cList = [
             # [0.796, 0.023, 0.023], #red
             [1, 0, 0], #red
             # [0.607, 0.505, 0.192], #gold, brown
             # [0.219, 0.552, 0.066], #green
             [0.133, 0.545, 0.133], # green
             # [0.419, 0.098, 0.882], #purple
             [0.462, 0.368, 0.078], # dark brown
             [0, 0, 1], #blue
             # [0.054, 0.741, 0.768], #cyan
             [1, 0, 1], #violet
             [0.690, 0.247, 0.811]] #magenta
    # cList = [[0.862, 0.078, 0.235],
    #          [1, 0.549, 0],
    #          # [1, 0.549, 0],
    #          [0.580, 0, 0.827],
    #          # [0.211, 0.662, 0.058],
    #          [0, 0.501, 0],
    #          [0.254, 0.411, 0.882]]
    color_dict = {}
    i = 0
    for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
        color_dict[lbl] = cList[i]
        i += 1
    return color_dict

def get_marker_dict(links_stat_dirs, labels):
    experiment_comp_paths = links_stat_dirs.split(',')
    ls = labels.split(',')
    marker = itertools.cycle(('o', 'v', 's','*', 'd'))
    marker_dict = {}
    i = 0
    for links_stat_dir, lbl in itertools.izip(experiment_comp_paths, ls):
        marker_dict[lbl] = marker.next()
    return marker_dict


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_group_ip_bw_dict(bw_map_file):
    ip_bw_map = {}
    try:
        with open(bw_map_file) as f:
            lines = [x.strip('\n') for x in f.readlines()]
        index = 0
        for line in lines:
            ip_bw = line.split('-')
            ip = ip_bw[0].strip()
            bw = ip_bw[1].strip()
            ip_bw_map[ip] =  bw
    except Exception as e:
        print ('error in reading file named ' + bw_map_file)
        sys.exit(1)
    return ip_bw_map


def get_group_bw_dict(bw_map_file):
    ip_bw_map = {}
    try:
        with open(bw_map_file) as f:
            lines = [x.strip('\n') for x in f.readlines()]
        index = 0
        for line in lines:
            ip_bw = line.split('-')
            ip = ip_bw[0].strip()
            bw = ip_bw[1].strip()
            ip_bw_map[index] = [ip, bw]
            index = int(index) + 1
    except Exception as e:
        print ('error in reading file named ' + bw_map_file)
        sys.exit(1)
    return ip_bw_map


def compute_distances(network_g, source, targets):
    distance_map = {}
    path_length_dict = nx.single_source_dijkstra_path_length(network_g, source)
    for t in targets:
        distance_map[t] = path_length_dict[t]
    # print distance_map
    return distance_map

def get_exp_dists(network_g, source, targets):
    exp_dist = {}
    distance_map = compute_distances(network_g, source, targets)
    for n, d in distance_map.iteritems():
        exp_val = np.exp(-1 * float(d))
        exp_dist[n] = float(exp_val)
    return exp_dist


def get_porbs(node_dist_map, node_list=None):
    nodes = []
    probs = []
    total = 0.0
    if node_list is None:
        node_keys = node_dist_map.keys()
    else:
        node_keys = node_list
    for n in node_keys:
        d = node_dist_map[n]
        total += d
    for n in node_keys:
        nodes.append(n)
        d = node_dist_map[n]
        probs.append(float(d)/float(total))
    return  nodes, probs

