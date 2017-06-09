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



class MulticastEvent(object):

    def __init__(self, tree_id, tree_s, tree_bw, receiver_node, ev_time, event_type='join'):
        self.t_index = tree_id
        self.t_source = tree_s
        self.t_bandwidth = tree_bw
        self.recv_node = receiver_node
        self.ev_type = event_type
        self.ev_time = ev_time