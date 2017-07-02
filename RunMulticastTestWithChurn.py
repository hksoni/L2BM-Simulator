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


import multiprocessing
import json
import os
from DynamicSteinerTree import *
from DynamicSteinerTreeLB import *
from L2BM import *
from MulticastEvent import *
from AlgoExecutionThread import AlgoExecutionThread as AlgoExecThread
from Heap import *
import  copy


def main_fn(network_g, group_numbers, run_start, run_stop, mean_run_stats=False):
    'Number of multicast groups'
    parallel_runs = 4
    runs = range(int(run_start), int(run_stop), 1)
    group_numbers_list = group_numbers.split(',')
    bandwidths = [1, 2, 5, 9, 12, 25, 50, 80]
    mean_sender_inter_arrival = 0.001
    mean_receiver_inter_arrival = 5
    mean_reception_duration = 75.0
    run_scheds = True
    link_eliminate_bw_filter_value = '1000000'
    theta_inits = [10, 20, 30, 40, 50, 60]
    candidate_nodes_data = filter(lambda (n, d): d['type'] == 'switch', network_g.nodes(data=True))
    candidate_nodes = [i[0] for i in candidate_nodes_data]
    #results will be stored at -> save_results_path/<algo-name>/#groups/run<no>/
    # save_results_path = '/root/churn/congested-va-bw-10-100'
    save_results_path = utils.working_dir+'/churn-congested/'
    sched_path = utils.working_dir+'/churn-congested/'
    th_list = []
    for num_groups in group_numbers_list:
        no_of_groups = int(num_groups)
        for run_num in runs:
            print 'No of Groups:'+str(no_of_groups)+',Run No.:'+str(run_num)
            sched_file_path = None
            if run_scheds:
                sched_file_path = sched_path+str(no_of_groups)+'-run'+str(run_num)
                if os.path.isfile(sched_file_path) is False:
                    sched_file_path = None
            group_bandwidths = np.random.choice(bandwidths, no_of_groups, replace=True)
            candidate_nodes_t = copy.deepcopy(candidate_nodes)
            t = multiprocessing.Process(target=execute_run, args=(
                network_g.copy(), link_eliminate_bw_filter_value, no_of_groups, theta_inits, mean_sender_inter_arrival,
                mean_receiver_inter_arrival, mean_reception_duration, candidate_nodes_t, group_bandwidths, run_num,
                save_results_path, mean_run_stats, sched_file_path))
            th_list.append(t)
            t.start()
            if len(th_list) == parallel_runs:
                for th in th_list:
                    th.join()
                del th_list[:]


def execute_run(network_graph, link_eliminate_bw_filter_value, no_of_groups, theta_inits, mean_sender_inter_arrival,
                mean_receiver_inter_arrival, mean_reception_duration, candidate_nodes, group_bandwidths, run_number,
                result_path, mean_run_stats, sched_file_path=None):
    events_list = None
    if sched_file_path is None:
        events_list, tree_objls_ls = create_sched_heap_congested(
            network_graph, no_of_groups, theta_inits, mean_sender_inter_arrival, mean_receiver_inter_arrival,
            mean_reception_duration, candidate_nodes, group_bandwidths, run_number, result_path)
    else:
        events_list, tree_objls_ls = get_sched_heap(sched_file_path, no_of_groups, theta_inits)
    algo_names = []
    algo_names.append('dst')
    algo_names.append('dst-lb')
    for theta in theta_inits:
        algo_names.append('l2bm-'+str(theta))
    tree_thread_obj_ls= []
    for al, tree_objls in itertools.izip(algo_names, tree_objls_ls):
        nw_g= network_graph.copy()
        # print algo_names
        algo_exec_th_obj = AlgoExecThread(al, nw_g, link_eliminate_bw_filter_value, events_list, tree_objls,
                                          result_path, no_of_groups, run_number, mean_run_stats)
        algo_exec_th_obj.start()
        tree_thread_obj_ls.append(algo_exec_th_obj)
    for th_obj in tree_thread_obj_ls:
        th_obj.join()


def create_sched_heap_congested(network_graph, no_of_groups, theta_inits, mean_sender_inter_arrival, mean_receiver_inter_arrival,
                                mean_reception_duration, candidate_nodes, group_bandwidths,
                                run_number, result_path):
    source = np.random.choice(candidate_nodes)
    c_nodes = copy.deepcopy(candidate_nodes)
    c_nodes.remove(source)
    node_dist_map = utils.get_exp_dists(network_graph, source, c_nodes)
    events_list, tree_objls_ls = create_sched_heap(
        theta_inits, no_of_groups, mean_sender_inter_arrival, mean_receiver_inter_arrival, mean_reception_duration,
        group_bandwidths, node_dist_map, no_nodes_in_source_pool=len(c_nodes)/3, no_nodes_in_pool=len(c_nodes),
        initial_group_size=15.0, churn_time_duration=500)
    # ev_path = '/user/hsoni/home/qos-multicast-compile/simulation/temp/70-run72'
    # events_list, tree_objls_ls = get_sched_heap(ev_path, no_of_groups)
    utils.mkdir(result_path)
    # dst_ls, dst_lb_ls, l2bm_10_ls, l2bm_30_ls, l2bm_40_ls, l2bm_50_ls, l2bm_60_ls = tree_objs
    f = open(result_path+"/"+str(no_of_groups)+'-run'+str(run_number),"w")
    for r,e in events_list:
        f.writelines(json.dumps(e.__dict__)+'\n')
    f.close()
    return events_list, tree_objls_ls


def create_sched_heap(theta_inits, no_of_groups, mean_sender_inter_arrival, mean_receiver_inter_arrival,
                                mean_reception_duration, group_bandwidths, node_weights_map,
                                no_nodes_in_source_pool=10, no_nodes_in_pool=20, initial_group_size=5.0,
                                churn_time_duration=300):
    ev_heap = Heap()
    ev_list = []
    dst_obj_ls = []
    dst_lb_obj_ls = []
    l2bm_theta_obj_dict = {}
    for theta in theta_inits:
        l2bm_theta_obj_dict[theta] = []
    event_index = 0

    nodes, probs = utils.get_porbs(node_weights_map)
    group_node_pool = np.random.choice(nodes, no_nodes_in_pool, replace=False, p=probs).tolist()

    source_node_pool = np.random.choice(nodes, no_nodes_in_source_pool, p=probs).tolist()
    source_nodes_pool, source_nodes_pool_probs = utils.get_porbs(node_weights_map, source_node_pool)
    sources = np.random.choice(source_nodes_pool, no_of_groups, p=source_nodes_pool_probs)
    sender_inter_arrival_time = np.random.exponential(mean_sender_inter_arrival, no_of_groups)
    sender_arrival_time = np.cumsum(sender_inter_arrival_time)
    lambda_r = 1.0/float(mean_receiver_inter_arrival)
    mu_r = 1.0/float(mean_reception_duration)

    for n in range(0,no_of_groups):
        print_lengths = []
        bw = group_bandwidths[n]

        recv_nodes, recv_nodes_probs = utils.get_porbs(node_weights_map, group_node_pool)
        initial_receivers = np.random.choice(recv_nodes, int(initial_group_size), replace=False,
                                             p=recv_nodes_probs).tolist()
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
        return_count = 0
        while time < float(churn_time_duration)+r_a and N > 0:
            ran1 = np.random.uniform()
            time_lambda = (-1.0) * np.log(float(ran1))/lambda_r
            ran2 = np.random.uniform()
            time_mu = (-1.0) * np.log(float(ran2))/(mu_r*N)
            # print 'time_lambda', time_lambda, 'time_mu', time_mu
            dt = min(time_lambda, time_mu)
            ev_time = time + dt
            print_lengths.append(len(existing_nodes))
            if time_lambda < time_mu and len(recv_nodes) > len(existing_nodes):
                N += 1
                # if len(recv_nodes) == len(existing_nodes):
                #     print 'lengths - ', print_lengths
                #     print 'All nodes are receiving and one more receiver to select to join'
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
                print ' birth and death time event at the same time: lambda:'+ str(time_lambda)+' mu: '+str(time_mu)
                if return_count == 20:
                    print 'error: schedule generation failed for no of groups:'+ str(no_of_groups)
                    return
                return_count += 1
                continue
            time = ev_time
            ev_heap.push(ev_time, event_index, me)
            event_index += 1
        dst = DynamicSteinerTree(source,bw)
        dst_lb = DynamicSteinerTreeLB(source,bw)
        dst_obj_ls.append(dst)
        dst_lb_obj_ls.append(dst_lb)
        for theta in theta_inits:
            l2bm_theta_obj_dict[theta].append(L2BM(source, bw, int(theta)))
    while not ev_heap.is_empty():
        r_a, item_id, ev = ev_heap.pop()
        ev_list.append((r_a, ev))
    tree_objls_ls = [dst_obj_ls, dst_lb_obj_ls]
    for theta in theta_inits:
        tree_objls_ls.append(l2bm_theta_obj_dict[theta])
    return ev_list, tree_objls_ls



def select_uniq_node(pool_nodes, existing_nodes_dict, pool_nodes_probs=None):
    while True:
        n = np.random.choice(pool_nodes, p=pool_nodes_probs)
        if n in existing_nodes_dict:
            continue
        return n


def get_sched_heap(file_name, no_of_groups, theta_inits):
    l2bm_theta_obj_dict = {}
    dst_obj_ls = []
    dst_lb_obj_ls = []
    for theta in theta_inits:
        l2bm_theta_obj_dict[theta] = []
    e_list = []
    bw_list = {}
    source_list = {}
    print 'Reading event schedule file named - '+file_name
    lines = [line.rstrip('\n') for line in open(file_name)]
    for l in lines:
        m = eval(l)
        me = MulticastEvent(int(m['t_index']), m['t_source'], int(m['t_bandwidth']), m['recv_node'],
                            float(m['ev_time']), m['ev_type'])
        r_a = float(m['ev_time'])
        bw_list[int(m['t_index'])] = int(m['t_bandwidth'])
        source_list[int(m['t_index'])] = m['t_source']
        e_list.append((r_a, me))
    for n in range(0,no_of_groups):
        dst = DynamicSteinerTree(source_list[n],bw_list[n])
        dst_lb = DynamicSteinerTreeLB(source_list[n],bw_list[n])
        for theta in theta_inits:
            l2bm_theta_obj_dict[theta].append(L2BM(source_list[n], bw_list[n], int(theta)))
        dst_obj_ls.append(dst)
        dst_lb_obj_ls.append(dst_lb)
    tree_objls_ls = [dst_obj_ls, dst_lb_obj_ls]
    for theta in theta_inits:
        tree_objls_ls.append(l2bm_theta_obj_dict[theta])
    return e_list, tree_objls_ls



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_dot", help="network topology in DOT format")
    parser.add_argument("--numbers_groups", help="multicast groups in CSV(for more than one)")
    parser.add_argument("--run_start_num", help="run start")
    parser.add_argument("--run_stop_num", help="run stop")

    args = parser.parse_args()
    num_groups = args.numbers_groups
    if args.network_dot is None:
        args.network_dot = utils.working_dir+"/internet2-al2s.dot"
    if args.numbers_groups  is None:
        # num_groups = '100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000'
        num_groups = '5000,6000,7000,8000,9000,10000,11000,12000,13000,14000,15000,16000,' \
                     '17000,18000,19000,20000'
    run_start = args.run_start_num
    run_stop = args.run_stop_num
    if args.run_start_num is None or args.run_stop_num is None:
        run_start = 0
        run_stop = 1
    network_g = nx.read_dot(args.network_dot)
    mean_run_stats = True
    main_fn(network_g, num_groups, run_start, run_stop, mean_run_stats)
