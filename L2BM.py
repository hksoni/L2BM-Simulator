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
import networkx as nx
import sys
import argparse
from MulticastTree import *
from Heap import *
# from heapq_showtree import show_tree
# from heapq_heapdata import data



class L2BM(MulticastTree):

    def __init__(self, sn, bw_dm, theta, increment=10):
        super(L2BM, self).__init__(sn, bw_dm)
        self.theta_init=theta
        self.theta_increment=increment

    def add_node(self, nw_topo_g, rn):
        # print 'adding node ' + rn
        next = {}
        distance = Heap()
        visited = []
        # for [n,d] in nw_topo_g.nodes(data=True):
        #     distance.push(sys.maxint, n, sys.maxint)
        distance.push(0, rn, sys.maxint)
        return self.threshold_based_bfs(nw_topo_g, rn, self.theta_init,
                                 visited, distance, next)


    def threshold_based_bfs(self, nw_topo_g, rn, threshold,
                            visited_nodes, distance_heap, next):
        theta = threshold
        # print 'length - '+str(len(visited_nodes))
        tree_node_found = False
        nn = rn
        pruned_heap = Heap()
        # distance_heap.print_heap_items()
        while not distance_heap.is_empty():
            dist, nn, path_max_util = distance_heap.pop()
            if nn in visited_nodes:
                continue
            visited_nodes.append(nn)
            # print 'Q pop nn -> '+ nn
            if nn in self.tree.nodes():
                tree_node_found = True
                # print 'tree node found'
                break
            for u,n,d in nw_topo_g.in_edges(nn, data=True):
                if u in visited_nodes:
                    continue
                bw_cap = int(d['capacity'])
                bw_cons = float(bw_cap) - (float(d['bandwidth']) - float(self.bandwidth))
                bw_util = float(bw_cons)*100/float(bw_cap)
                if bw_util > 100.0:
                    continue
                new_dist_u = dist + 1
                new_path_max_util = max(path_max_util, bw_util)
                if bw_util < float(theta):
                    # print u + '->'+nn
                    # print ''
                    dist_u_max_util = distance_heap.get_key(u)
                    if dist_u_max_util is not None:
                        dist_u, u_max_util = dist_u_max_util
                        if (dist_u < new_dist_u) or (dist_u == new_dist_u and new_path_max_util > u_max_util):
                            continue
                    distance_heap.push(new_dist_u, u, new_path_max_util)
                    next[u] = nn
                else:
                    # print 'put in pruned queue: ' + str(nn)
                    dist_u_max_util_pru_hp = pruned_heap.get_key(nn)
                    if dist_u_max_util_pru_hp is None:
                        pruned_heap.push(dist, nn, path_max_util)
        if tree_node_found:
            # print 'tree node found ' + str(nn)
            self.leaf_nodes.append(rn)
            i = 0
            while nn != rn and i<len(nw_topo_g.nodes()):
                ns = next[nn]
                self.tree.add_edge(nn, ns)
                bw = int(nw_topo_g.edge[nn][ns][0]['bandwidth'])
                nw_topo_g.edge[nn][ns][0]['bandwidth'] = bw-int(self.bandwidth)
                data = nw_topo_g.edge[nn][ns][0]['bandwidth']
                # print str(nn) + '->' +str(ns)
                nn = ns
                i = i+1
            if i >= len(nw_topo_g.nodes()):
                print 'error'
                print "Existing Tree "+str(self.tree.nodes())
                print "receiver" +  rn
                exit(1)
            return tree_node_found
        else:
            theta = float(theta) + float(self.theta_increment)
            if not pruned_heap.is_empty():
                # print 'recursive call'
                for x in pruned_heap.get_items():
                    visited_nodes.remove(x)
                return self.threshold_based_bfs(nw_topo_g, rn, theta,
                                                visited_nodes, pruned_heap, next)
            else:
                # print 'tree node not found'
                return False


def main(network_g):
    dst1 = L2BM('vs1', 6, 10, 10)
    dst2 = L2BM('vs1', 5, 10, 10)
    dst1.add_node(network_g, 'vs2')
    dst1.add_node(network_g, 'vs9')
    dst1.add_node(network_g, 'vs10')

    print 'tree1'
    dst1.print_tree()

    # dst1.remove_node(network_g, 'vs4')
    # dst1.remove_node(network_g, 'vs6')
    # dst1.remove_node(network_g, 'vs3')

    print 'tree2'
    dst2.add_node(network_g, 'vs13')
    dst2.add_node(network_g, 'vs11')
    dst2.add_node(network_g, 'vs3')
    dst2.print_tree()
    # dst.add_node(network_g, 'vs9')

    # dst1.remove_node(network_g, 'vs4')
    # dst1.remove_node(network_g, 'vs6')
    # dst1.remove_node(network_g, 'vs3')

    # print network_g.edges(data=True)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_dot", help="network topology in DOT format")
    args = parser.parse_args()
    if args.network_dot is None:
        args.network_dot = "/user/hsoni/home/internet2-al2s.dot"
    network_g = nx.read_dot(args.network_dot)
    main(network_g)