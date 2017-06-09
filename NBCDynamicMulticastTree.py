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
import itertools
from heapq import heappush, heappop
import argparse
import networkx as nx
import numpy as np
from os import system as cmd
import os.path
import sys
import json


def main(vnet_f):
    virt_g = nx.read_dot(vnet_f)
    # nodes_ct = nx.betweenness_centrality(virt_g, normalized=False)
    # print(nodes_ct)
    print(str("----------------------"))


    # G = virt_g

    G=nx.Graph()
    G.add_path([0, 1, 2, 4])
    G.add_path([0, 5, 3, 4])
    G.add_path([5, 2])
    T=nx.Graph()
    # T.add_path(['vs1', 'vs2'])
    T.add_path([0, 1])
    group_members = [3]
    # print(virt_g.node['vs1'])


    # print("\n"+str(input_trees))
    all_pair_all_shortest_paths_dict = create_all_pair_all_shortest_path_map(G)
    # print all_pair_all_shortest_paths_dict
    # total = 0
    # for t in input_trees:
    #     ts = get_tree_size(G, t[0], t[1], all_pair_all_shortest_paths_dict)
    #     total += ts
    #     # print (str(t[0]) + "-" +str(t[1]) + ":"+str(ts))
    #     print (str(ts))
    # print(total)


def get_test_trees(G):
    dir_path = '/home/hsoni/qos-multicast-compile/exp3-vabw-25-75/dst/'
    file_name = 'group_mem_ia_rd.txt'
    g_dir_names = [30]
    run_dir_names = ['run3']
    # g_dir_names = [10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    # run_dir_names = ['run1','run2','run3','run4','run5']
    input_trees = []
    for gds in g_dir_names:
        for r in run_dir_names:
            group_mem_ia_rd = {}
            file_full_path = dir_path+str(gds)+'/'+r+'/'+file_name
            group_mem_ia_rd_str_key = eval(open(file_full_path, 'r').read())
            for k,v in group_mem_ia_rd_str_key.iteritems():
                group_mem_ia_rd[int(k)] = v
            for k,v in group_mem_ia_rd.iteritems():
                mems = []
                for h, i,r in v:
                    ns = G.neighbors(h)
                    mems.append(ns[0])
                input_trees.append([mems[0],mems[1:]])
                # print(str(mems[0]) +' '+str(mems[1:]))

def get_tree_size(network_g, src_node, group_members, all_pair_all_shortest_paths_dict):
    T=nx.Graph()
    T.add_node(src_node)
    for mn in group_members:
        node_score_d = find_node_score(network_g, T, mn, all_pair_all_shortest_paths_dict)
        path_for_new_node = find_highest_node_score_shortest_path_to_tree(
            network_g, T, node_score_d, mn, all_pair_all_shortest_paths_dict)
        T.add_path(path_for_new_node)
    return len(T.edges())


def find_node_score(network_g, partial_tree_g, new_node, all_pair_all_shortest_paths_dict):
    node_score_dict = {}
    for nn in network_g.nodes():
        node_score_dict[nn] = 0
    node_list = partial_tree_g.nodes()
    node_list.append(new_node)
    for tn in partial_tree_g.nodes():
        # for nn, nd in network_g.nodes(data=True):
        for nn in network_g.nodes():
            # if tn == nn or nd['type']=='host':
            if tn == nn:
                continue
            path_nodes_set = set()
            for p in all_pair_all_shortest_paths_dict[(tn,nn)]:
            # for p in nx.all_shortest_paths(network_g,source=tn,target=nn):
                for np in p:
                    if np == tn or np == nn:
                        continue
                    path_nodes_set.add(np)
            # print str(tn) + ' ' +str(nn)
            # print path_nodes_set
            for n in path_nodes_set:
                node_score_dict[n] += 1
    # print node_score_dict
    return node_score_dict


def create_all_pair_all_shortest_path_map(g):
    st_all_shortest_paths = {}
    for s in g.nodes():
        for t in g.nodes():
            paths = []
            for path in nx.all_shortest_paths(g,source=s,target=t):
                paths.append(path)
            st_all_shortest_paths[(s,t)] = paths
    # print st_all_shortest_paths
    return  st_all_shortest_paths


def find_highest_node_score_shortest_path_to_tree(network_g, partial_tree_g,
                                                  node_score_dict, new_node, all_pair_all_shortest_paths_dict):
    current_min_path_length = sys.maxsize
    max_node_score = 0
    nearest_node_paths = None
    max_node_score_shortest_path = None
    for tn in partial_tree_g.nodes():
        paths = all_pair_all_shortest_paths_dict[(new_node,tn)]
        if len(paths) > 0 and len(paths[0]) < current_min_path_length:
            current_min_path_length = len(paths[0])
            nearest_node = tn
            nearest_node_paths = paths
    # print str('***********************')
    # print ('minimum length paths')
    # print nearest_node_paths
    # print str('***********************')
    for p in nearest_node_paths:
        temp_score = 0
        for np in p:
            temp_score += node_score_dict[np]
        if temp_score > max_node_score:
            max_node_score = temp_score
            max_node_score_shortest_path = p
    # print ('Path added: '+ str(max_node_score_shortest_path))
    return max_node_score_shortest_path



def get_steiner_tree_size(network_g, src_node, group_members, all_pair_all_shortest_paths_dict):
    T=nx.Graph()
    T.add_node(src_node)
    for mn in group_members:
        path_length = sys.maxsize
        shortest_path_to_tree = None
        for tn in T.nodes():
            paths_for_mn = all_pair_all_shortest_paths_dict[(tn, mn)]
            if path_length > len(paths_for_mn[0]):
                path_length  = len(paths_for_mn[0])
                shortest_path_to_tree = paths_for_mn[0]
        T.add_path(shortest_path_to_tree)
    return len(T.edges())




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--virtual_net", help="DOT file specifying experimental network topology")
    args = parser.parse_args()
    main(args.virtual_net)
    # main(args.physical_net, args.virtual_net, args.mapping, args.bw_map,
    #      args.exe_schedule, args.sender_schedule, args.num_groups)

# #---test example
# G=nx.Graph()
# G.add_path([0, 1, 2, 4])
# G.add_path([0, 5, 3, 4])
# G.add_path([5, 2])
# print([p for p in nx.all_shortest_paths(G,source=0,target=4)])
# print(nx.betweenness_centrality(G, normalized=False))
# #---test example