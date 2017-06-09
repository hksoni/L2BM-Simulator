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
import os, errno, sys
import numpy as np
import networkx as nx
from multiprocessing import Process
from multiprocessing.pool import ThreadPool


class AlgoExecutionThread(Process):

    def __init__(self, algo_n, nw_g, e_l, tree_objs, no_receiver_to_wait, re_path, no_of_g, run_num):
        self.algo_name = algo_n
        self.n_g= nw_g
        self.events = e_l
        self.tree_objs = tree_objs
        self.receiver_accept = 0
        self.bw_accept = 0
        self.bw = 0
        self.tn = 0
        self.bn_l = []
        self.link_stats_churn = []
        self.no_of_groups = no_of_g
        self.run_number = run_num
        self.result_path = re_path
        super(AlgoExecutionThread, self).__init__()

    def run(self):
        bw_accept_map_ratio = {}
        total_bw_request_map = {}
        total_bw_request = 0
        total_receivers_request = 0
        bw_accept_map = {}
        r_p = self.result_path+'/'+self.algo_name+'/'+str(self.no_of_groups)+'/run'+str(self.run_number)
        self.mkdir(r_p)
        # print 'execute_sched_on_algo'
        link_caps = filter(lambda x: x!='1000', nx.get_edge_attributes(self.n_g, 'capacity').values())
        np_link_caps = np.array(link_caps, dtype=float)
        # log_str_file = open(r_p+'/log-res.txt',"w")
        for r_a, event in self.events:
            log_str = json.dumps(event.__dict__)
            ret = True
            # print json.dumps(event.__dict__)
            tree = self.tree_objs[event.t_index]
            if event.ev_type == 'join':
                # print 'add node - > '+event.recv_node
                ret = tree.add_node(self.n_g, event.recv_node)
                total_bw_request += tree.bandwidth
                total_receivers_request += 1
                if total_bw_request_map.has_key(str(tree.bandwidth)):
                    total_bw_request_map[str(tree.bandwidth)] += 1
                else:
                    total_bw_request_map[str(tree.bandwidth)] = 1
                if ret:
                    self.receiver_accept += 1
                    self.bw_accept += tree.bandwidth
                    # log_str += '--- Success'+'\n'
                    if bw_accept_map.has_key(str(tree.bandwidth)):
                        bw_accept_map[str(tree.bandwidth)] += 1
                    else:
                        bw_accept_map[str(tree.bandwidth)] = 1

                else:
                    ''
                    # log_str += '--- Fail'+'\n'
            elif event.ev_type == 'leave':
                # print 'add node - > '+event.recv_node
                ret = tree.remove_node(self.n_g, event.recv_node)
                # log_str += '*************'+'\n'
            else:
                print 'unknown event'
            # log_str_file.writelines(log_str)
            link_bws = filter(lambda x: x!='1000', nx.get_edge_attributes(self.n_g, 'bandwidth').values())
            # print link_bws
            np_link_bws = np.array(link_bws, dtype=float)
            np_utils = np.true_divide(np.subtract(np_link_caps, np_link_bws), np_link_caps)
            self.link_stats_churn.append([float(r_a), np.mean(np_utils), np.std(np_utils), np.max(np_utils),
                                          self.no_links_gt_lu(np_utils, 0.60),
                                          self.no_links_gt_lu(np_utils, 0.70),
                                          self.no_links_gt_lu(np_utils, 0.80),
                                          self.no_links_gt_lu(np_utils, 0.90)])
        # log_str_file.close()
        for bw, num in total_bw_request_map.iteritems():
            bw_accept_map_ratio[str(bw)] = str(float(bw_accept_map[str(bw)]) / float(num))
        with open(r_p+'/churn-link-bw-stats.txt',"w") as link_stat_file:
            link_stat_file.writelines(json.dumps(self.link_stats_churn))
        with open(r_p+'/churn-other-results.txt',"w") as results_file:
            results_file.writelines("receiver-requests-accept=" + str(self.receiver_accept) + '\n')
            results_file.writelines("bandwidth-accept=" + str(self.bw_accept) + '\n')
            results_file.writelines("bandwidth-accept-map="+json.dumps(bw_accept_map)+'\n')
            results_file.writelines("receivers-accept-ratio=" +
                                    str(float(self.receiver_accept)/float(total_receivers_request)) + '\n')
            results_file.writelines("bandwidth-accept-ratio=" +
                                    str(float(self.bw_accept)/float(total_bw_request)) + '\n')
            results_file.writelines("bandwidth-accept-map-ratio="+json.dumps(bw_accept_map_ratio)+'\n')

        # return (self.n_g, self.receiver_accept, self.bw_accept, bn_l, tn)

    def mkdir(self, r):
        try:
            os.makedirs(r)
        except OSError as exc:
            if exc.errno == errno.EEXIST and os.path.isdir(r):
                pass
            else:
                raise


    def no_links_gt_lu(self, np_array, x):
        num = (np_array >= x).sum()
        return float(num)






def squareNumber(n):
    return (n[0] * n[1] , n[0] ,n[1])

# function to be mapped over
def calculateParallel(numbers, threads=2):
    po = ThreadPool(threads)
    results = po.map(squareNumber, numbers)
    po.close()
    po.join()
    return results

if __name__ == "__main__":
    numbers = [(1,3), (2,3), (3,3), (4,3),(5,3)]
    # numbers = [1,3, 2,3, 3,3, 4,3, 5,3]
    squaredNumbers = calculateParallel(numbers, 4)
    for a,b,c in squaredNumbers:
        print(str(b) + '*' +str(c)+ ' = '+str(a))