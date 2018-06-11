# coding: utf-8
from __future__ import absolute_import
from __future__ import print_function

import time


def time_cost(func):
    def __time_cost(*args, *kwargs):
        start = time.time()
        

