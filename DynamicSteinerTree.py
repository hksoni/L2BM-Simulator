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
import numpy as np
from MulticastTree import *
from Heap import *
# from heapq_showtree import show_tree
# from heapq_heapdata import data



class DynamicSteinerTree(MulticastTree):

    def __init__(self, sn, bw_dm):
        super(DynamicSteinerTree, self).__init__(sn, bw_dm)
        # self.leaf_nodes = []
        # self.source_node = sn
        # self.tree = nx.DiGraph()
        # self.bandwidth = bw_dm
        # self.tree.add_node(sn)

    def add_node(self, nw_topo_g, rn):
        # print 'DynamicSteinerTree:Source:'+self.source_node+' adding node ' + rn
        visited = []
        tree_node_found = False
        distance = Heap()
        next = {}
        distance.push(0, rn)
        nn = rn
        while not distance.is_empty():
            dist, nn, dnn = distance.pop()
            # print 'Q pop nn -> '+ nn
            if nn in visited:
                continue
            visited.append(nn)
            if nn in self.tree.nodes():
                tree_node_found = True
                # print 'tree node found'
                break
            for u,n,d in nw_topo_g.in_edges(nn, data=True):
                if (u in visited) or (int(d['bandwidth']) < int(self.bandwidth)):
                    continue
                # print u + '->'+nn
                new_dist_u = dist + 1
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
                if  bw-int(self.bandwidth) < 0:
                    print self.source_node, rn, self.bandwidth
                data = nw_topo_g.edge[nn][ns][0]['bandwidth']
                # print str(nn) + '->' +str(ns)
                nn = ns
        return tree_node_found


def main(network_g):
    dst = DynamicSteinerTree('vs1', 50)
    dst.add_node(network_g, 'vs6')
    dst.add_node(network_g, 'vs9')
    dst.add_node(network_g, 'vs7')
    dst.add_node(network_g, 'vs2')
    for u,v,x in dst.tree.edges(data=True):
        print u, v, network_g.get_edge_data(u,v)
    dst.print_tree()
    # print dst.branch_nodes()
    dst.remove_node(network_g, 'vs7')
    dst.remove_node(network_g, 'vs9')
    dst.remove_node(network_g, 'vs6')
    dst.remove_node(network_g, 'vs2')
    dst.print_tree()
    # dst.print_tree()
    for u,v,x in dst.tree.edges(data=True):
        print u, v, network_g.get_edge_data(u,v)

    i = 0
    for u,v,x in network_g.edges(data=True):
        if x['bandwidth']!= '1000':
            print u,v, x
            i +=1
    print i
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_dot", help="network topology in DOT format")
    args = parser.parse_args()
    if args.network_dot is None:
        args.network_dot = "/user/hsoni/home/internet2-al2s.dot"
    network_g = nx.read_dot(args.network_dot)
    main(network_g)