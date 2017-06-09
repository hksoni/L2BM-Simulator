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
from datetime import time

import networkx as nx
import sys
import argparse
import numpy as np
from  time import clock
import multiprocessing
import json
import Utils as utils
import os
import errno
from DynamicSteinerTree import *
from DynamicSteinerTreeLB import *
from L2BM import *
from MulticastEvent import *
from AlgoExecutionThread import AlgoExecutionThread as AlgoExecThread
from Heap import *
import  copy
from multiprocessing.dummy import Pool as ThreadPool



def main(network_g):
    'Number of multicast groups'
    parallel_runs = 7
    runs = range(100)
    # group_numbers_list = [10, 100]
    # group_numbers_list = [20, 90]
    # group_numbers_list = [30, 80]
    # group_numbers_list = [40, 70]
    group_numbers_list = [50, 60]

    bandwidths = [2, 5, 8]
    # bandwidths = [2]
    mean_receiver_inter_arrival = 5
    mean_reception_duration = 75.0
    no_of_receiver_per_group = 100
    candidate_nodes_data = filter(lambda (n, d): d['type'] == 'switch', network_g.nodes(data=True))
    candidate_nodes = [i[0] for i in candidate_nodes_data]
    #results will be stored at -> save_results_path/<algo-name>/#groups/run<no>/
    # save_results_path = '/root/churn/congested-va-bw-10-100'
    save_results_path = utils.working_dir+'/qos-multicast-compile/simulation/churn-congested/'
    th_list = []
    for no_of_groups in group_numbers_list:
        for run_num in runs:
            print 'No of Groups:'+str(no_of_groups)+',Run No.:'+str(run_num)
            group_bandwidths = np.random.choice(bandwidths, no_of_groups, replace=True)
            candidate_nodes_t = copy.deepcopy(candidate_nodes)
            t = multiprocessing.Process(target=execute_run, args=(network_g.copy(), no_of_groups, mean_receiver_inter_arrival,
                                                           mean_reception_duration, no_of_receiver_per_group,
                                                           candidate_nodes_t, group_bandwidths, run_num,
                                                           save_results_path))
            th_list.append(t)
            t.start()
            if len(th_list) == parallel_runs:
                for th in th_list:
                    th.join()
                del th_list[:]
    # for no_of_groups in group_numbers_list:
    #     for run_num in runs:
    #         print 'No of Groups:'+str(no_of_groups)+',Run No.:'+str(run_num)
    #         group_bandwidths = np.random.choice(bandwidths, no_of_groups, replace=True)
    #         candidate_nodes_t = copy.deepcopy(candidate_nodes)
    #         execute_run(network_g.copy(), no_of_groups, mean_receiver_inter_arrival,
    #                     mean_reception_duration, no_of_receiver_per_group,
    #                     candidate_nodes_t, group_bandwidths, run_num, save_results_path)


def execute_run(network_graph, no_of_groups, mean_receiver_inter_arrival, mean_reception_duration,
                no_of_receiver_per_group, candidate_nodes, group_bandwidths, run_number, result_path):
    algo_names = ['dst', 'dst-lb', 'l2bm-10', 'l2bm-40', 'l2bm-60']
    # algo_names = ['dst']
    expected_no_receivers = int(mean_reception_duration/mean_receiver_inter_arrival)
    # print expected_no_receivers
    # events_list, tree_objls_ls = create_sched_heap(no_of_groups, mean_receiver_inter_arrival,
    #                                                mean_reception_duration, no_of_receiver_per_group,
    #                                                candidate_nodes, group_bandwidths)
    source = np.random.choice(candidate_nodes)
    c_nodes = copy.deepcopy(candidate_nodes)
    c_nodes.remove(source)
    node_dist_map = utils.get_exp_dists(network_g, source, c_nodes)
    events_list, tree_objls_ls = create_sched_heap_congested(
        no_of_groups, mean_receiver_inter_arrival, mean_reception_duration, no_of_receiver_per_group,
        group_bandwidths, node_dist_map, no_nodes_in_source_pool=len(c_nodes)/3, no_nodes_in_pool=len(c_nodes), initial_group_size=20.0,
        churn_time_duration=300)
    # ev_path = '/user/hsoni/home/qos-multicast-compile/simulation/temp/70-run72'
    # events_list, tree_objls_ls = get_sched_heap(ev_path, no_of_groups)
    mkdir(result_path)
    # dst_ls, dst_lb_ls, l2bm_10_ls, l2bm_30_ls, l2bm_40_ls, l2bm_50_ls, l2bm_60_ls = tree_objs
    f = open(result_path+"/"+str(no_of_groups)+'-run'+str(run_number),"w")
    for r,e in events_list:
        f.writelines(json.dumps(e.__dict__)+'\n')
    f.close()
    tree_thread_obj_ls= []
    for al, tree_objls in itertools.izip(algo_names, tree_objls_ls):
        nw_g= network_graph.copy()
        # print algo_names
        algo_exec_th_obj = AlgoExecThread(al, nw_g, events_list, tree_objls, expected_no_receivers,
                                          result_path, no_of_groups, run_number)
        algo_exec_th_obj.start()
        tree_thread_obj_ls.append(algo_exec_th_obj)
    for th_obj in tree_thread_obj_ls:
        th_obj.join()


def create_sched_heap_congested(no_of_groups, mean_receiver_inter_arrival, mean_reception_duration,
                                no_of_receiver_per_group, group_bandwidths, node_weights_map,
                                no_nodes_in_source_pool=10, no_nodes_in_pool=20,
                                initial_group_size=5.0, churn_time_duration=300):
    ev_heap = Heap()
    ev_list = []
    dst_obj_ls = []
    dst_lb_obj_ls = []
    l2bm_10_obj_ls = []
    l2bm_40_obj_ls = []
    l2bm_60_obj_ls = []
    event_index = 0


    nodes, probs = utils.get_porbs(node_weights_map)
    group_node_pool = np.random.choice(nodes, no_nodes_in_pool, replace=False, p=probs).tolist()
    # print group_node_pool

    source_node_pool = np.random.choice(nodes, no_nodes_in_source_pool, p=probs).tolist()
    source_nodes_pool, source_nodes_pool_probs = utils.get_porbs(node_weights_map, source_node_pool)
    sources = np.random.choice(source_nodes_pool, no_of_groups, p=source_nodes_pool_probs)
    sender_inter_arrival_time = np.random.exponential(3, no_of_groups)
    sender_arrival_time = np.cumsum(sender_inter_arrival_time)

    lambda_r = 1.0/float(mean_receiver_inter_arrival)
    mu_r = 1.0/float(mean_reception_duration)

    # pool_nodes, pool_nodes_probs = utils.get_porbs(node_weights_map, group_node_pool)
    for n in range(0,no_of_groups):
        print_lengths = []
        bw = group_bandwidths[n]

        recv_nodes, recv_nodes_probs = utils.get_porbs(node_weights_map, group_node_pool)
        initial_receivers = np.random.choice(recv_nodes, int(initial_group_size), replace=False,
                                             p=recv_nodes_probs).tolist()
        # source = initial_receivers.pop(0)
        source = sources[n]
        initial_recv_inter_arrival = np.random.exponential(float(mean_receiver_inter_arrival),
                                                           int(initial_group_size))
        existing_nodes = {}
        existing_nodes[source] = source
        r_a = sender_arrival_time[n]

        for init_recv_node, inter_arrival in itertools.izip(initial_receivers, initial_recv_inter_arrival):
            r_a += float(inter_arrival)
            # print id, r_a, r_d
            existing_nodes[init_recv_node] = init_recv_node
            me_a = MulticastEvent(n, source, int(bw), init_recv_node, float(r_a), 'join')
            ev_heap.push(r_a, event_index, me_a)
            event_index += 1

        time = r_a
        N = int(initial_group_size)
        # print time, float(churn_time_duration)+r_a
        while time < float(churn_time_duration)+r_a and N > 0:
            ran1 = np.random.uniform()
            time_lambda = (-1.0) * np.log(float(ran1))/lambda_r
            ran2 = np.random.uniform()
            time_mu = (-1.0) * np.log(float(ran2))/(mu_r*N)
            # print 'time_lambda', time_lambda, 'time_mu', time_mu
            dt = min(time_lambda, time_mu)
            ev_time = time + dt
            print_lengths.append(len(existing_nodes))
            if time_lambda < time_mu:
                N += 1
                if len(recv_nodes) == len(existing_nodes):
                    print 'lengths - ', print_lengths
                    print '------- Error -------'
                join_node = select_uniq_node(recv_nodes, existing_nodes,
                                             pool_nodes_probs=recv_nodes_probs)

                existing_nodes[join_node] = join_node
                me = MulticastEvent(n, source, int(bw), join_node, float(ev_time), 'join')
            elif time_lambda > time_mu:
                N -= 1
                leave_node = select_uniq_node(existing_nodes.keys(), {})
                existing_nodes.pop(leave_node)
                me = MulticastEvent(n, source, int(bw), leave_node, float(ev_time), 'leave')
            else:
                continue
            time = ev_time
            ev_heap.push(ev_time, event_index, me)
            event_index += 1
            # if i%10 == 0:
            #     exit(0)
            # i += 1
        # print print_lengths
        # exit(0)
        dst = DynamicSteinerTree(source,bw)
        dst_lb = DynamicSteinerTreeLB(source,bw)
        l2bm_10 = L2BM(source, bw, 10)
        l2bm_40 = L2BM(source, bw, 40)
        l2bm_60 = L2BM(source, bw, 60)
        dst_obj_ls.append(dst)
        dst_lb_obj_ls.append(dst_lb)
        l2bm_10_obj_ls.append(l2bm_10)
        l2bm_40_obj_ls.append(l2bm_40)
        l2bm_60_obj_ls.append(l2bm_60)
    while not ev_heap.is_empty():
        # print 'sorting'
        r_a, item_id, ev = ev_heap.pop()
        ev_list.append((r_a, ev))
    # return ev_list, [dst_obj_ls]
    return ev_list, [dst_obj_ls, dst_lb_obj_ls, l2bm_10_obj_ls, l2bm_40_obj_ls, l2bm_60_obj_ls]



def select_uniq_node(pool_nodes, existing_nodes_dict, pool_nodes_probs=None):
    while True:
        n = np.random.choice(pool_nodes, p=pool_nodes_probs)
        if n in existing_nodes_dict:
            continue
        return n


# def create_sched_heap(no_of_groups, mean_receiver_inter_arrival, mean_reception_duration,
#                       no_of_receiver_per_group, candidate_nodes, group_bandwidths):
#     ev_heap = Heap()
#     ev_list = []
#     dst_obj_ls = []
#     dst_lb_obj_ls = []
#     l2bm_10_obj_ls = []
#     l2bm_40_obj_ls = []
#     l2bm_60_obj_ls = []
#     event_index = 0
#     sources = np.random.choice(candidate_nodes, no_of_groups)
#     for n in range(0,no_of_groups):
#         bw = group_bandwidths[n]
#         receiver_inter_arrival_time = np.random.exponential(mean_receiver_inter_arrival,
#                                                             no_of_receiver_per_group)
#         reception_duration = np.random.exponential(mean_reception_duration,
#                                                    no_of_receiver_per_group)
#         candidate_nodes.remove(sources[n])
#         # print receiver_inter_arrival_time
#         receiver_arrival_time = np.cumsum(receiver_inter_arrival_time)
#         receiver_departure_time = np.add(receiver_arrival_time, reception_duration)
#         # print receiver_arrival_time
#         receivers = range(no_of_receiver_per_group)
#         # receivers = np.random.choice(candidate_nodes, no_of_receiver_per_group+1, replace=False).tolist()
#         g_e_id = 0
#         g_ev_heap = Heap()
#         id_node = {}
#         # print len(candidate_nodes)
#         for id, r_a, r_d in itertools.izip(receivers, receiver_arrival_time, receiver_departure_time):
#             me_a = MulticastEvent(n, sources[n], int(bw), id, r_a, 'join')
#             me_d = MulticastEvent(n, sources[n], int(bw), id, r_d, 'leave')
#             g_arr_item_id = g_e_id + 1
#             g_e_id += 2
#             g_dep_item_id = g_e_id
#             g_ev_heap.push(r_a, g_arr_item_id, me_a)
#             g_ev_heap.push(r_d, g_dep_item_id, me_d)
#         while not g_ev_heap.is_empty():
#             r_a, item_id, ev = g_ev_heap.pop()
#             if ev.ev_type == 'join':
#                 r_n = np.random.choice(candidate_nodes)
#                 id_node[ev.recv_node] = r_n
#                 ev.recv_node = r_n
#                 candidate_nodes.remove(r_n)
#             elif ev.ev_type == 'leave':
#                 node = id_node[ev.recv_node]
#                 ev.recv_node = node
#                 candidate_nodes.append(node)
#             else:
#                 print(' -- unknown event -- ')
#             ev_heap.push(r_a, event_index, ev)
#             event_index += 1
#         candidate_nodes.append(sources[n])
#         dst = DynamicSteinerTree(sources[n],bw)
#         dst_lb = DynamicSteinerTreeLB(sources[n],bw)
#         l2bm_10 = L2BM(sources[n], bw, 10)
#         l2bm_40 = L2BM(sources[n], bw, 40)
#         l2bm_60 = L2BM(sources[n], bw, 60)
#         dst_obj_ls.append(dst)
#         dst_lb_obj_ls.append(dst_lb)
#         l2bm_10_obj_ls.append(l2bm_10)
#         l2bm_40_obj_ls.append(l2bm_40)
#         l2bm_60_obj_ls.append(l2bm_60)
#         # for r_n, r_a, r_d in itertools.izip(receivers, receiver_arrival_time, receiver_departure_time):
#         #     me_a = MulticastEvent(n, source, int(bw), r_n, r_a, 'join')
#         #     me_d = MulticastEvent(n, source, int(bw), r_n, r_d, 'leave')
#         #     arr_item_id = event_index+1
#         #     event_index = event_index+2
#         #     dep_item_id = event_index
#         #     # print item_id
#         #     ev_heap.push(r_a, arr_item_id, me_a)
#         #     ev_heap.push(r_d, dep_item_id, me_d)
#     while not ev_heap.is_empty():
#         # print 'sorting'
#         r_a, item_id, ev = ev_heap.pop()
#         ev_list.append((r_a, ev))
#     return ev_list, [dst_obj_ls]
#     # return ev_list, [dst_obj_ls, dst_lb_obj_ls, l2bm_10_obj_ls, l2bm_40_obj_ls, l2bm_60_obj_ls]



def get_sched_heap(file_name, no_of_groups):
    ev_heap = Heap()
    ev_list = []
    dst_obj_ls = []
    dst_lb_obj_ls = []
    l2bm_10_obj_ls = []
    l2bm_30_obj_ls = []
    l2bm_40_obj_ls = []
    l2bm_50_obj_ls = []
    l2bm_60_obj_ls = []
    event_index = 0
    #####
    e_list = []
    bw_list = {}
    source_list = {}
    lines = [line.rstrip('\n') for line in open(file_name)]
    for l in lines:
        m = eval(l)
        me = MulticastEvent(int(m['t_index']), m['t_source'], int(m['t_bandwidth']), m['recv_node'],
                            float(m['ev_time']), m['ev_type'])
        r_a = float(m['ev_time'])
        bw_list[int(m['t_index'])] = int(m['t_bandwidth'])
        source_list[m['t_index']] = m['t_source']
        e_list.append((r_a, me))
    #####
    for n in range(0,no_of_groups):
        dst = DynamicSteinerTree(source_list[n],bw_list[n])
        dst_lb = DynamicSteinerTreeLB(source_list[n],bw_list[n])
        l2bm_10 = L2BM(source_list[n], bw_list[n], 10)
        l2bm_40 = L2BM(source_list[n], bw_list[n], 40)
        l2bm_60 = L2BM(source_list[n], bw_list[n], 60)
        dst_obj_ls.append(dst)
        dst_lb_obj_ls.append(dst_lb)
        l2bm_10_obj_ls.append(l2bm_10)
        l2bm_40_obj_ls.append(l2bm_40)
        l2bm_60_obj_ls.append(l2bm_60)
        # for r_n, r_a, r_d in itertools.izip(receivers, receiver_arrival_time, receiver_departure_time):
        #     me_a = MulticastEvent(n, source, int(bw), r_n, r_a, 'join')
        #     me_d = MulticastEvent(n, source, int(bw), r_n, r_d, 'leave')
        #     arr_item_id = event_index+1
        #     event_index = event_index+2
        #     dep_item_id = event_index
        #     # print item_id
        #     ev_heap.push(r_a, arr_item_id, me_a)
        #     ev_heap.push(r_d, dep_item_id, me_d)
    while not ev_heap.is_empty():
        # print 'sorting'
        r_a, item_id, ev = ev_heap.pop()
        ev_list.append((r_a, ev))
    # return e_list, [dst_obj_ls]
    return ev_list, [dst_obj_ls, dst_lb_obj_ls, l2bm_10_obj_ls, l2bm_40_obj_ls, l2bm_60_obj_ls]



def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_dot", help="network topology in DOT format")
    args = parser.parse_args()
    if args.network_dot is None:
        args.network_dot = utils.working_dir+"/internet2-al2s.dot"
    network_g = nx.read_dot(args.network_dot)
    main(network_g)
