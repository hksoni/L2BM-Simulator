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
from __builtin__ import float

import json
import os
import errno
from DynamicSteinerTree import *
from DynamicSteinerTreeLB import *
from L2BM import *
from MulticastEvent import *
from Heap import *
from multiprocessing import Pool
import Utils as utils

def main(network_g):
    'Number of multicast groups'
    runs = range(200, 500, 1)
    group_numbers_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # group_numbers_list = [100]
    bandwidths = [2, 5, 8]
    # bandwidths = [2]
    mean_receiver_inter_arrival = 5.0
    no_of_receiver_per_group = 13
    candidate_nodes_data = filter(lambda (n, d): d['type'] == 'switch', network_g.nodes(data=True))
    candidate_nodes = [i[0] for i in candidate_nodes_data]
    #results will be stored at -> save_results_path/<algo-name>/#groups/run<no>/
    save_results_path = utils.working_dir+'/qos-multicast-compile/simulation/wo-churn/congested-va-bw/'
    for no_of_groups in group_numbers_list:
        for run_num in runs:
            print 'no_of_groups:'+str(no_of_groups) +',run_num:'+str(run_num)
            group_bandwidths = np.random.choice(bandwidths, no_of_groups, replace=True)
            execute_run(network_g, no_of_groups, mean_receiver_inter_arrival, no_of_receiver_per_group,
                        candidate_nodes, group_bandwidths, run_num, save_results_path)


def execute_run(network_graph, no_of_groups, mean_receiver_inter_arrival,
                no_of_receiver_per_group, candidate_nodes,
                group_bandwidths, run_number, result_path):
    mkdir(result_path)
    algo_names = ['dst', 'dst-lb', 'l2bm-10', 'l2bm-40', 'l2bm-60']
    source = np.random.choice(candidate_nodes)
    dists_map = utils.get_exp_dists(network_g, source, candidate_nodes)
    nodes, probs = utils.get_porbs(dists_map)
    events_list, dst_ls, dst_lb_ls, l2bm_10_ls, l2bm_40_ls, l2bm_60_ls = \
    create_sched_heap_congested_traffic(no_of_groups, mean_receiver_inter_arrival, no_of_receiver_per_group,
                      group_bandwidths, nodes, probs)
    # save_result_run_path = result_path+'/'+a_n+'/'+str(no_of_groups)+'/run'+str(run_number)
    launch_run(result_path, events_list, network_graph, dst_ls, dst_lb_ls, l2bm_10_ls,
               l2bm_40_ls, l2bm_60_ls, algo_names, no_of_groups, run_number)


def grid_main(network_g):
    'Number of multicast groups'
    runs = np.arange(1, 21, 1)
    group_numbers_list = [30, 40, 50, 60, 70]
    save_res_path = utils.working_dir+'/qos-multicast-compile/sim-exp3-vabw-30-70/'
    schedule_path = utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/dcbr-10-10/'
    bw_map_file = utils.working_dir+'/qos-multicast-compile/exp3-vabw-25-70-LLDMs/ip-bw-qos-mapping-va.txt'
    for no_of_groups in group_numbers_list:
        for run_num in runs:
            print 'no_of_groups:'+str(no_of_groups) +',run_num:'+str(run_num)
            run_schedule_path = schedule_path+'/'+str(no_of_groups)+'/run'+str(run_num)
            # execute_controller_exec_seq(network_g, bw_map_file, no_of_groups, run_num,
            #                             save_res_path, run_schedule_path)
            execute_grid_run(network_g, bw_map_file, no_of_groups, run_num, save_res_path, run_schedule_path)


def execute_grid_run(network_graph, bw_map_file, no_of_groups, run_number, result_path, scheds_path):
    mkdir(result_path)
    algo_names = ['dst', 'dst-lb', 'l2bm-10', 'l2bm-40', 'l2bm-60']
    events_list, dst_ls, dst_lb_ls, l2bm_10_ls, l2bm_40_ls, l2bm_60_ls = \
        get_grid_experiment_inputs(scheds_path, bw_map_file)
    # save_result_run_path = result_path+'/'+a_n+'/'+str(no_of_groups)+'/run'+str(run_number)
    launch_run(result_path, events_list, network_graph, dst_ls, dst_lb_ls, l2bm_10_ls,
               l2bm_40_ls, l2bm_60_ls, algo_names, no_of_groups, run_number)


def execute_controller_exec_seq(network_graph, bw_map_file, no_of_groups, run_number, result_path, scheds_path):
    mkdir(result_path)
    algo_names = ['dst', 'dst-lb', 'l2bm-10', 'l2bm-40', 'l2bm-60']
    events_list, dst_ls, dst_lb_ls, l2bm_10_ls, l2bm_40_ls, l2bm_60_ls = \
        get_controller_exec_seq(scheds_path, bw_map_file)
    # save_result_run_path = result_path+'/'+a_n+'/'+str(no_of_groups)+'/run'+str(run_number)
    launch_run(result_path, events_list, network_graph, dst_ls, dst_lb_ls, l2bm_10_ls,
               l2bm_40_ls, l2bm_60_ls, algo_names, no_of_groups, run_number)


def create_sched_heap(no_of_groups, mean_receiver_inter_arrival, no_of_receiver_per_group,
                      group_bandwidths, candidate_nodes, nodes_weight=None):
    ev_heap = Heap()
    ev_list = []
    dst_obj_ls = []
    dst_lb_obj_ls = []
    l2bm_10_obj_ls = []
    l2bm_40_obj_ls = []
    l2bm_60_obj_ls = []
    event_index = 0
    sender_inter_arrival_time = np.random.exponential(3, no_of_groups)
    sender_arrival_time = np.cumsum(sender_inter_arrival_time)
    for n in range(0,no_of_groups):
        receiver_inter_arrival_time = np.random.exponential(mean_receiver_inter_arrival,
                                                            no_of_receiver_per_group)

        # print receiver_inter_arrival_time
        receiver_arrival_time = np.add(np.cumsum(receiver_inter_arrival_time), sender_arrival_time[int(n)])
        # print receiver_arrival_time
        receivers = np.random.choice(candidate_nodes, no_of_receiver_per_group+1, replace=False).tolist()
        source = receivers.pop(0)
        bw = group_bandwidths[n]
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
        for r_n, r_a in itertools.izip(receivers, receiver_arrival_time):
            me = MulticastEvent(n, source, int(bw), r_n, r_a, 'join')
            item_id = event_index
            event_index = event_index + 1
            # print item_id
            ev_heap.push(r_a, item_id, me)
    while not ev_heap.is_empty():
        r_a, item_id, ev = ev_heap.pop()
        ev_list.append((r_a, ev))
    return ev_list, dst_obj_ls, dst_lb_obj_ls, l2bm_10_obj_ls, l2bm_40_obj_ls, l2bm_60_obj_ls


def create_sched_heap_congested_traffic(no_of_groups, mean_receiver_inter_arrival, no_of_receiver_per_group,
                                        group_bandwidths, candidate_nodes, nodes_weight=None):
    ev_heap = Heap()
    ev_list = []
    dst_obj_ls = []
    dst_lb_obj_ls = []
    l2bm_10_obj_ls = []
    l2bm_40_obj_ls = []
    l2bm_60_obj_ls = []
    event_index = 0
    sender_inter_arrival_time = np.random.exponential(3, no_of_groups)
    sender_arrival_time = np.cumsum(sender_inter_arrival_time)

    # concentrated_nodes = np.random.choice(candidate_nodes, no_of_receiver_per_group+1, replace=False).tolist()
    for n in range(0,no_of_groups):

        receiver_inter_arrival_time = np.random.exponential(mean_receiver_inter_arrival,
                                                            no_of_receiver_per_group)

        # print receiver_inter_arrival_time
        receiver_arrival_time = np.add(np.cumsum(receiver_inter_arrival_time), sender_arrival_time[int(n)])
        # print receiver_arrival_time
        receivers = np.random.choice(candidate_nodes, no_of_receiver_per_group+1, replace=False, p=nodes_weight).tolist()
        source = receivers.pop(0)
        bw = group_bandwidths[n]
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
        for r_n, r_a in itertools.izip(receivers, receiver_arrival_time):
            me = MulticastEvent(n, source, int(bw), r_n, r_a, 'join')
            item_id = event_index
            event_index = event_index + 1
            # print item_id
            ev_heap.push(r_a, item_id, me)
    while not ev_heap.is_empty():
        r_a, item_id, ev = ev_heap.pop()
        ev_list.append((r_a, ev))
    return ev_list, dst_obj_ls, dst_lb_obj_ls, l2bm_10_obj_ls, l2bm_40_obj_ls, l2bm_60_obj_ls



def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_grid_experiment_inputs(run_location, bw_map_file):
    group_mem_ia_rd_dict = {}
    group_id_ip_bw_dict = {}
    sender_sched_f = run_location+'/group_sender_sched.txt'
    group_bw_f = bw_map_file
    group_mem_ia_rd_f = run_location+'/group_mem_ia_rd.txt'
    group_mem_ia_rd_str_key = eval(open(group_mem_ia_rd_f, 'r').read())
    for k,v in group_mem_ia_rd_str_key.iteritems():
        group_mem_ia_rd_dict[int(k)] = v
    group_sender_inter_arrival_ls = np.array(eval(open(sender_sched_f, 'r').read()), dtype=float)
    group_id_ip_bw_str_key = utils.get_group_bw_dict(group_bw_f)
    for k,v in group_id_ip_bw_str_key.iteritems():
        group_id_ip_bw_dict[int(k)] = int(v[1])/1000
        # print group_id_ip_bw_dict[int(k)]
        # group_id_ip_bw_dict[int(k)] = int(2)
    group_sender_arrivals = np.cumsum(group_sender_inter_arrival_ls)
    ev_heap = Heap()
    ev_list = []
    event_index = 0
    # print 'range(len(group_mem_ia_rd_dict))', range(len(group_mem_ia_rd_dict))
    dst_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    dst_lb_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    l2bm_10_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    l2bm_40_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    l2bm_60_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    for k,v in group_mem_ia_rd_dict.iteritems():
        sender_arrival_tiime = group_sender_arrivals[k]
        source = v[0][0]
        del v[0]
        bw = group_id_ip_bw_dict[int(k)]
        dst = DynamicSteinerTree(source,bw)
        dst_lb = DynamicSteinerTreeLB(source,bw)
        l2bm_10 = L2BM(source, bw, 10)
        l2bm_40 = L2BM(source, bw, 40)
        l2bm_60 = L2BM(source, bw, 60)
        dst_obj_ls[int(k)] = dst
        dst_lb_obj_ls[int(k)] = dst_lb
        l2bm_10_obj_ls[int(k)] = l2bm_10
        l2bm_40_obj_ls[int(k)] = l2bm_40
        l2bm_60_obj_ls[int(k)] = l2bm_60
        cumsum_arrival_time = sender_arrival_tiime
        for r_info in v:
            recv, i_a, rd = r_info
            cumsum_arrival_time += i_a
            me = MulticastEvent(int(k), source, int(bw), recv, cumsum_arrival_time, 'join')
            item_id = event_index
            event_index += 1
            ev_heap.push(cumsum_arrival_time, item_id, me)
    while not ev_heap.is_empty():
        r_a, item_id, ev = ev_heap.pop()
        ev_list.append((r_a, ev))
    return ev_list, dst_obj_ls, dst_lb_obj_ls, l2bm_10_obj_ls, l2bm_40_obj_ls, l2bm_60_obj_ls


def get_controller_exec_seq(run_location, bw_map_file):
    group_mem_ia_rd_dict = {}
    group_id_ip_bw_dict = {}
    group_bw_f = bw_map_file
    ev_list = []
    group_id_ip_bw_str_key = utils.get_group_bw_dict(group_bw_f)
    for k,v in group_id_ip_bw_str_key.iteritems():
        # group_id_ip_bw_dict[int(k)] = int(v[1])/1000
        group_id_ip_bw_dict[int(k)] = int(2)
    controller_seq_f = run_location+'/controller.seq'
    group_mem_ia_rd_f = run_location+'/group_mem_ia_rd.txt'
    group_mem_ia_rd_str_key = eval(open(group_mem_ia_rd_f, 'r').read())
    for k,v in group_mem_ia_rd_str_key.iteritems():
        group_mem_ia_rd_dict[int(k)] = v
    dst_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    dst_lb_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    l2bm_10_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    l2bm_40_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    l2bm_60_obj_ls = [None for _ in range(len(group_mem_ia_rd_dict))]
    for k,v in group_mem_ia_rd_dict.iteritems():
        source = v[0][0]
        bw = group_id_ip_bw_dict[int(k)]
        dst = DynamicSteinerTree(source,bw)
        dst_lb = DynamicSteinerTreeLB(source,bw)
        l2bm_10 = L2BM(source, bw, 10)
        l2bm_40 = L2BM(source, bw, 40)
        l2bm_60 = L2BM(source, bw, 60)
        dst_obj_ls[int(k)] = dst
        dst_lb_obj_ls[int(k)] = dst_lb
        l2bm_10_obj_ls[int(k)] = l2bm_10
        l2bm_40_obj_ls[int(k)] = l2bm_40
        l2bm_60_obj_ls[int(k)] = l2bm_60
    try:
        with open(controller_seq_f) as f:
            lines = [x.strip('\n') for x in f.readlines()]
            index = 0
        for line in lines:
            ips = line.split(" IP:")
            m_ip = ips[0].strip()
            r_ip = ips[1].strip()
            receiver = 'vs'+r_ip.split('.')[-1]
            tree_id = int(m_ip.split('.')[-1]) - 1
            source_node = group_mem_ia_rd_dict[tree_id][0][0]
            bw = group_id_ip_bw_dict[tree_id]
            me = MulticastEvent(tree_id, source_node, int(bw), receiver, 0, 'join')
            ev_list.append((0, me))
    except Exception as e:
        print ('error in reading file named ' + controller_seq_f)
        sys.exit(1)
    return ev_list, dst_obj_ls, dst_lb_obj_ls, l2bm_10_obj_ls, l2bm_40_obj_ls, l2bm_60_obj_ls


# def get_group_bw_dict(bw_map_file):
#     ip_bw_map = {}
#     try:
#         with open(bw_map_file) as f:
#             lines = [x.strip('\n') for x in f.readlines()]
#         index = 0
#         for line in lines:
#             ip_bw = line.split('-')
#             ip = ip_bw[0].strip()
#             bw = ip_bw[1].strip()
#             ip_bw_map[index] = [ip, bw]
#             index = int(index) + 1
#     except Exception as e:
#         print ('error in reading file named ' + bw_map_file)
#         sys.exit(1)
#     return ip_bw_map



def launch_run(result_path, events_list, network_graph, dst_ls, dst_lb_ls, l2bm_10_ls,
               l2bm_40_ls, l2bm_60_ls, algo_names, no_of_groups, run_number):
    f = open(result_path+str(no_of_groups)+'-'+str(run_number)+str('-ev_list.txt'),"w")
    for r,e in events_list:
        f.writelines(json.dumps(e.__dict__)+'\n')
    f.close()
    e_list = []
    # lines = [line.rstrip('\n') for line in open('event_list.txt')]
    # for l in lines:
    #     m = eval(l)
    #     me = MulticastEvent(m['t_index'], m['t_source'], m['t_bandwidth'], m['recv_node'], m['ev_time'], m['ev_type'])
    #     r_a = m['ev_time']
    #     e_list.append((r_a, me))
    nw_g_dst = network_graph.copy()
    nw_g_dst_lb = network_graph.copy()
    nw_g_l2bm_10 = network_graph.copy()
    nw_g_l2bm_40 = network_graph.copy()
    nw_g_l2bm_60 = network_graph.copy()
    pool = Pool(4)
    # args = [('l2bm-10', nw_g_l2bm_10, events_list, l2bm_10_ls)]
    args = [('dst', nw_g_dst, events_list, dst_ls),
            ('dst_lb', nw_g_dst_lb, events_list, dst_lb_ls),
            ('l2bm-10', nw_g_l2bm_10, events_list, l2bm_10_ls),
            ('l2bm-40', nw_g_l2bm_40, events_list, l2bm_40_ls),
            ('l2bm-60', nw_g_l2bm_60, events_list, l2bm_60_ls)]
    results = pool.map(execute_sched_on_algo, args)
    pool.close()
    pool.join()
    for a_n, r in itertools.izip(algo_names, results):
        g, rr, br, bn, tn, bw_map, receivers_accept_ratio, bw_ratio, bw_map_ratio, total_bw_request_map = r
        r_p = result_path+'/'+a_n+'/'+str(no_of_groups)+'/run'+str(run_number)
        mkdir(r_p)
        edge_str = ''
        for u,v,d in g.edges(data=True):
            if str(u).startswith('h') or str(v).startswith('h'):
                continue
            edge_str+= '('+u+','+v+')='+str(float(d['capacity'])-float(d['bandwidth']))+','
        with open(r_p+'/link-bw-stats.txt',"w") as link_stat_file:
            link_stat_file.writelines(edge_str)
        with open(r_p+'/other-results.txt',"w") as results_file:
            results_file.writelines("receiver-requests-accept="+str(rr)+'\n')
            results_file.writelines("bandwidth-accept="+str(br)+'\n')
            results_file.writelines("branch-nodes="+str(bn)+'\n')
            results_file.writelines("total-tree-nodes="+str(tn)+'\n')
            results_file.writelines("bandwidth-accept-map="+json.dumps(bw_map)+'\n')
            results_file.writelines("receivers-accept-ratio="+str(receivers_accept_ratio)+'\n')
            results_file.writelines("bandwidth-accept-ratio="+str(bw_ratio)+'\n')
            results_file.writelines("bandwidth-accept-map-ratio="+json.dumps(bw_map_ratio)+'\n')
            results_file.writelines("total-bandwidth-request-map="+json.dumps(total_bw_request_map)+'\n')



# [algo_name, nw_g, e_l, trees_obj_list]
def execute_sched_on_algo(args):
    total_bw_request_map = {}
    total_receivers_request = 0
    total_bw_request = 0.0
    algo_name = args[0]
    nw_g = args[1]
    e_l = args[2]
    tree_objs = args[3]
    receiver_accept = 0
    bw_accept = 0
    bw_accept_map = {}
    bw_accept_map_ratio = {}
    bw = 0
    # print 'execute_sched_on_algo'
    for r_a, event in e_l:
        ret = True
        # print json.dumps(event.__dict__)
        if event.ev_type == 'join':
            tree = tree_objs[event.t_index]
            # print 'add node - > '+event.recv_node
            ret = tree.add_node(nw_g, event.recv_node)
            bw = tree.bandwidth
            total_receivers_request += 1
            total_bw_request += float(bw)
            if total_bw_request_map.has_key(str(tree.bandwidth)):
                total_bw_request_map[str(tree.bandwidth)] += 1
            else:
                total_bw_request_map[str(tree.bandwidth)] = 1
            if ret:
                receiver_accept += 1
                bw_accept += float(bw)
                if bw_accept_map.has_key(str(bw)):
                    bw_accept_map[str(bw)] += 1
                else:
                    bw_accept_map[str(bw)] = 1
    for bw, num in total_bw_request_map.iteritems():
        bw_accept_map_ratio[str(bw)] = float(bw_accept_map[str(bw)]) / float(num)
    tn = 0
    bn_l = []
    for t in tree_objs:
        bn_l.append(t.branch_nodes())
        tn += t.tree_nodes()
    return (nw_g, receiver_accept, bw_accept, bn_l, tn, bw_accept_map,
            float(receiver_accept)/float(total_receivers_request),
            bw_accept/total_bw_request, bw_accept_map_ratio, total_bw_request_map)



def test_fn():
    l = [1,2,3]
    a, b, c = l
    print a,  b, c
    print l


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_dot", help="network topology in DOT format")
    args = parser.parse_args()
    if args.network_dot is None:
        # args.network_dot = "/user/hsoni/home/internet2-al2s.dot"
        args.network_dot = utils.working_dir+'/internet2-al2s.dot'
    network_g = nx.read_dot(args.network_dot)
    # get_grid_experiment_inputs('/user/hsoni/home/qos-multicast-compile/exp2-2mb-10-100/dst/')
    main(network_g)
    # grid_main(network_g)
    # get_controller_exec_seq('/home/hsoni/qos-multicast-compile/exp2-2mb-10-100/dst/100/run1/',
    #                         '/user/hsoni/home/qos-multicast-compile/exp2-2mb-10-100/ip-bw-qos-mapping.txt-const')