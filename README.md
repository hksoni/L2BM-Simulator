# L2BM-Simulator
This is a simulator to generate different kind of multicast traffic scenarios like concenrated traffic during prime-time or peak hours as explained in more details at https://team.inria.fr/diana/software/l2bm/.

# Features:
1. The simulator can create and store the multicast event schedule on disk for a given traffic scenario.
2. It can replay the given schedule.
3. It can optionally provide network statistics at the granularity of event.
4. It includes the necessary but basic scripts to share the simulator compute load on multiple nodes and collect results at a given node.


The simualator has been tested on Fedora Core 19 and above. 
Following command will help to install required software.

```
sudo dnf install -y numpy scipy python-matplotlib ipython python-pandas sympy python-nose atlas-devel
```
if `pyparsing` is not already installed and your matplotlib version throws error, use
```
pip install pyparsing
```
Python IDE [PyCharm](https://www.jetbrains.com/pycharm/?fromMenu) can be used to browse the local copy of the code.
`.idea` folder contains PyCharm project files. (Note: Remove the unavaliable project in the workspace.)

# Usage:
1. To run multicast without churn, use `python RunMulticastTest.py -h `
2. To run multicast with churn, use `python RunMulticastTestWithChurn.py -h `
3. To replay the schedules for tests executed on other testbed and, look at `grid_main` function in `RunMulticastTest.py`
4. To replay the schedules for tests executed on simulator, loot at `execute_run` function in `RunMulticastTest.py`
5. Setting `mean_run_stats=True` in `__main__` of `RunMulticastTestWithChurn.py` will disable the storing on utilization of all the links at each event. 
6. Use `PlotWithoutChurnStats.py` and `PlotChurnStats.py` to plot the graphs.
7. For superimposed plots between testbed and simulator experiments look at the `PlotGridSimCompareStats.py`.

# Results:
We have uploaded extended results as jpeg in the result folder. We are working on ipython notebook based setup for better and easy reproducibility of results.

