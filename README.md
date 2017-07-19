# L2BM-Simulator
This is a simulator to generate different kind of multicast traffic scenarios like concenrated traffic during prime-time or peak hours as explained in more details at https://team.inria.fr/diana/software/l2bm/.

Features:
1. The simulator can create and store the multicast event schedule on disk for a given traffic scenario.
2. It can replay the given schedule.
3. It can optionally provide network statistics at the granularity of event.
4. It includes the necessary but basic scripts to share the simulator compute load on multiple nodes and collect results at a given node.


The simualator has been tested on Fedora Core 19 and above. 
Following command will help to install required software.

sudo dnf install -y numpy scipy python-matplotlib ipython python-pandas sympy python-nose atlas-devel
if your matplotlib version throws error for pyparsing, use

pip install pyparsing

