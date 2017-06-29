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
from pexpect import pxssh
from pexpect import *
import pexpect
import json
import itertools
import time
from time import gmtime, strftime, localtime
import sys
import os
import io
import signal
import argparse
import threading
import networkx as nx
import subprocess, shlex
import numpy as np
from os import system as cmd
import os.path


node_list_g = ['parasilo-1', 'parasilo-2']

start_run = 0
end_run = 2


def get_node_list_from_file(node_list_file):
    if node_list_file is None:
        node_list = node_list_g
    else:
        with open(node_list_file) as f:
            node_list = f.readlines()
        node_list = [x.strip() for x in node_list]
    return  node_list


def main(id, node_list_file, cmd):
    node_list = get_node_list_from_file(node_list_file)
    process_dict = {}
    process_list = []
    closed_process_list = []
    ret = False
    for node in node_list:
        process = pexpect.spawnu('ssh -t  root' + '@' + node+' ' +cmd)
        process_dict[node] = process
        process_list.append(process)
    while ret is False:
        for node, p in process_dict.iteritems():
            # print node, vals
            if  not process_list:
                print 'process list empty'
                ret = True
            if p.isalive() is False and p not in closed_process_list:
                closed_process_list.append(p)
                process_list.remove(p)
                print 'Execution on node '+ node +' finished'


def main_node_cmd_list(id, node_cmd_list_file):
    node_cmd_list = get_node_list_from_file(node_cmd_list_file)
    process_dict = {}
    process_list = []
    closed_process_list = []
    ret = False
    for node_cmd in node_cmd_list:
        node, cmd = node_cmd.split()
        process = pexpect.spawnu('ssh -t  root' + '@' + node+' ' +cmd)
        process_dict[node] = process
        process_list.append(process)
    while ret is False:
        for node, p in process_dict.iteritems():
            # print node, vals
            if  not process_list:
                print 'process list empty'
                ret = True
            if p.isalive() is False and p not in closed_process_list:
                closed_process_list.append(p)
                process_list.remove(p)
                print 'Execution on node '+ node +' finished'



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_id", help="experiment_id_tag")
    parser.add_argument("--node_list_file", help="nodes")
    parser.add_argument("--node_cmd_list_file", help="nodes")
    args = parser.parse_args()
    if args.exp_id is None:
        args.exp_id = str(np.random.randint(0, 1000,1))
    if args.node_list_file is not None:
        main(args.exp_id, args.node_list_file, cmd)
    elif args.node_cmd_list_file is not None:
        main_node_cmd_list(id, args.node_cmd_list_file)