# L2BM-Simulator
This is a simulator to generate different kind of multicast traffic scenarios.
It create multicast even schedules based on time stamp and feeds it to the implemented algorithm.

It also contains scripts to plot the generated metrics.

Features:
1. The simulator can create and store the multicast event schedule on disk for a given traffic scenario.
2. It can replay the given schedule.
3. It can optionally provide network statistics at the granularity of event.
4. It includes the necessary but basic scripts to share the simulator compute load on multiple nodes and collect results at a given node.
