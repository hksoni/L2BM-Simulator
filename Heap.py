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
import sys
import heapq

class Heap(object):
    def __init__(self, initial=None):
        self.entry_finder = {}
        self.counter = itertools.count()
        self._heap = []
        self.REMOVED = '<removed>'
        self.no_of_removed = 0


    def remove_item(self, item):
        'Mark an existing task as REMOVED.  Raise KeyError if not found.'
        entry = self.entry_finder.pop(item)
        entry[2] = self.REMOVED
        self.no_of_removed = int(self.no_of_removed) + 1


    def push(self, key, item, data=None):
        if item in self.entry_finder:
            self.remove_item(item)
        count = next(self.counter)
        entry = [key, count, item, data]
        self.entry_finder[item] = entry
        heapq.heappush(self._heap, entry)


    def get_key(self, item):
        if item in self.entry_finder:
            [key, count, i, d] = self.entry_finder[item]
            return key, d
        return None


    def pop(self):
        'Remove and return the lowest priority task. Raise KeyError if empty.'
        while self._heap:
            key, count, item, data = heapq.heappop(self._heap)
            if item is not self.REMOVED:
                del self.entry_finder[item]
                return key, item, data
            self.no_of_removed -= 1
        raise KeyError('pop from an empty priority queue')


    def is_empty(self):
        if len(self._heap) == self.no_of_removed:
            return True
        return  False
    # ele = filter(lambda (key, count, i, d): i != self.REMOVED, self.entry_finder.iteritems())


    def get_items(self):
        return self.entry_finder.keys()


    def print_heap_items(self):
        print  self.entry_finder.keys()