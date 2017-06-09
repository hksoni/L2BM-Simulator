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
# from heapq_showtree import show_tree
# from heapq_heapdata import data



class MulticastTree(object):


    def __init__(self, sn, bw_dm):
        self.leaf_nodes = []
        self.source_node = sn
        self.tree = nx.DiGraph()
        self.bandwidth = bw_dm
        self.tree.add_node(sn)


    def print_tree(self):
        print '----------------------------'
        print self.tree.edges()
        print '----------------------------'


    def remove_node(self, nw_topo_g, rn):
        if rn not in self.leaf_nodes:
            return
        # print '----------------------------'
        node = rn
        # print 'removing node ' + node
        self.leaf_nodes.remove(rn)
        # print self.tree.edges()
        while self.tree.out_degree(node) == 0 \
                and node not in self.leaf_nodes \
                and node != self.source_node:
            u, n = self.tree.in_edges(node)[0]
            node = u
            # print 'Removing edges : '+u+"->"+n
            self.tree.remove_edge(u,n)
            self.tree.remove_node(n)
            bw = int(nw_topo_g.edge[u][n][0]['bandwidth'])
            nw_topo_g.edge[u][n][0]['bandwidth'] = bw+int(self.bandwidth)
        # print self.tree.edges()
        # print '----------------------------'


    def branch_nodes(self):
        # branch_nodes = 0
        out_degrees = self.tree.out_degree(self.tree.nodes())
        # print(out_degrees)
        branch_nodes_d = [i[0] for i in (filter(lambda (n, d): d>1, out_degrees.iteritems()))]
        # print(branch_nodes_d)
        # for n,d in out_degrees.iteritems():
        #     if d > 1:
        #         branch_nodes += 1
        return branch_nodes_d

    def tree_nodes(self):
        return self.tree.number_of_nodes()