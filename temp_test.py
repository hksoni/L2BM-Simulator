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
import matplotlib
matplotlib.use('Agg')

import scipy.stats as ss
import matplotlib
matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
import matplotlib.pyplot as plt
plt.style.use('classic')
from Utils import *


# def get_simulation_data(file_name, trunc_index=None):
#     # print file_name
#     with open(file_name) as f:
#         line = [x.strip('\n') for x in f.readlines()][0]
#         data = json.loads(line)
#         np_data = np.array(data, dtype=float)
#         np_data[:,4:8] *= 100
#         np_data[:,4:8] /= 102
#         if trunc_index is not None:
#             n_d = np.delete(np_data , trunc_index, 0)
#             return  n_d.tolist()
#         else:
#             return np_data.tolist()


if __name__ == "__main__":
    # print Utils.working_dir
    # print Utils.bw_map_file
    mean_sender_inter_arrival = 0.01
    no_of_groups = 20000
    sender_inter_arrival_time = np.random.exponential(mean_sender_inter_arrival, no_of_groups)
    sender_arrival_time = np.cumsum(sender_inter_arrival_time)
    print "---sender_arrival_time---"
    print sender_arrival_time



