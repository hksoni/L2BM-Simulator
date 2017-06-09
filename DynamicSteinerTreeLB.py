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
from Heap import *
from MulticastTree import *
# from heapq_showtree import show_tree
# from heapq_heapdata import data



class DynamicSteinerTreeLB(MulticastTree):

    def __init__(self, sn, bw_dm):
        super(DynamicSteinerTreeLB, self).__init__(sn, bw_dm)
        # self.leaf_nodes = []
        # self.source_node = sn
        # self.tree = nx.DiGraph()
        # self.bandwidth = bw_dm
        # self.tree.add_node(sn)

    def add_node(self, nw_topo_g, rn):
        # print 'adding node ' + rn
        tree_node_found = False
        distance = Heap()
        visited = []
        next = {}
        distance.push(0, rn)
        nn = rn
        while not distance.is_empty():
            dist, nn, dnn = distance.pop()
            # print 'Q pop nn -> '+ nn
            if nn in visited:
                continue
            if nn in self.tree.nodes():
                tree_node_found = True
                # print 'tree node found'
                break
            visited.append(nn)
            for u,n,d in nw_topo_g.in_edges(nn, data=True):
                if u in visited or int(d['bandwidth']) < int(self.bandwidth):
                    continue
                bw_cap = float(d['capacity'])
                # print 'bw_cap '+ str(bw_cap)
                bw_cons = float(bw_cap) - (float(d['bandwidth']) - float(self.bandwidth))
                # print 'bw_cons '+ str(bw_cons)
                bw_util = float(bw_cons)*100/float(bw_cap)
                new_dist_u = float(dist) + float(bw_util)
                dist_u_max_util = distance.get_key(u)
                if dist_u_max_util is not None:
                    dist_u, u_max_util = dist_u_max_util
                    if dist_u < new_dist_u:
                        continue
                distance.push(new_dist_u, u)
                next[u] = n
        if tree_node_found:
            self.leaf_nodes.append(rn)
            while nn != rn:
                ns = next[nn]
                self.tree.add_edge(nn, ns)
                bw = int(nw_topo_g.edge[nn][ns][0]['bandwidth'])
                nw_topo_g.edge[nn][ns][0]['bandwidth'] = bw-int(self.bandwidth)
                data = nw_topo_g.edge[nn][ns][0]['bandwidth']
                # print str(nn) + '->' +str(ns)
                nn = ns
        return tree_node_found


def main(network_g):
    dst1 = DynamicSteinerTreeLB('vs1', 10)
    dst2 = DynamicSteinerTreeLB('vs1', 110)
    dst1.add_node(network_g, 'vs4')
    dst1.add_node(network_g, 'vs6')
    dst1.add_node(network_g, 'vs3')
    dst1.print_tree()
    print 'tree 2'
    dst2.add_node(network_g, 'vs4')
    dst2.add_node(network_g, 'vs6')
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