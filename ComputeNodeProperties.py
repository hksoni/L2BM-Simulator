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
import argparse
import networkx as nx
import numpy as np
from os import system as cmd
import os.path



def main(vnet_f):
    virt_g = nx.read_dot(vnet_f)
    nodes_ct = nx.betweenness_centrality(virt_g, normalized=False)
    print(nodes_ct)
    print(str(""))
    G=nx.Graph()
    G.add_path([0, 1, 2, 4])
    G.add_path([0, 5, 3, 4])
    G.add_path([5, 2])
    print([p for p in nx.all_shortest_paths(G,source=0,target=4)])
    print(nx.betweenness_centrality(G, normalized=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--virtual_net", help="DOT file specifying experimental network topology")
    args = parser.parse_args()
    main(args.virtual_net)
    # main(args.physical_net, args.virtual_net, args.mapping, args.bw_map,
    #      args.exe_schedule, args.sender_schedule, args.num_groups)
