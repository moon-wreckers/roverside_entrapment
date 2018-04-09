#!/usr/bin/env python

"""
DecayFilter.py
Decay filter library.
"""

__version__     = "0.0.1"
__author__      = "David Qiu"
__email__       = "dq@cs.cmu.edu"
__website__     = "http://mrsdprojects.ri.cmu.edu/2017teami/"
__copyright__   = "Copyright (C) 2018, the Moon Wreckers. All rights reserved."

import numpy as np
from collections import deque


class DecayFilter(object):
    """
    Decay filter.
    """

    def __init__(self, d=0.95, wsize=32):
        """
        Initialize a decay filter.

        @param d The decay factor. (default: 0.95)
        @param wsize The filter window size. (default: 32)
        """

        super(DecayFilter, self).__init__()

        self.d = max(0.0, min(1.0, d))
        self.wsize = wsize

        # construct the filter queue
        self.q = deque(maxlen=wsize)

        # construct the decay template
        self.inv_h = [self.d ** i for i in range(self.wsize)]

    def append(self, x):
        """
        Append a data point.

        @param x The new data point to append.
        @return The current size of the filter queue.
        """

        self.q.append(np.array(x))

        return len(self.q)

    @property
    def filtered(self):
        """
        Get the filtered data.

        @return The filtered data.
        """

        n = len(self.q)

        if n < 1:
            return None

        sd = 0
        sx = 0
        for t in range(n):
            sx += self.q[n-1-t] * self.inv_h[t]
            sd += self.inv_h[t]

        ave = sx / sd

        return ave


def unit_test():
    print('UNIT TEST: DecayFilter')

    epsilon = 1e-9

    d     = 0.95
    wsize = 32

    print('\nTEST 1:')

    print('  - create decay filter (d=%f, wsize=%d)' % (d, wsize))
    filter0 = DecayFilter(d=d, wsize=wsize)

    print('  - get filtered data (filtered=%s)' % (filter0.filtered))
    assert(filter0.filtered is None)

    X = [10.5, 10.5, -100]

    print('  - append data (x=%f)' % (X[0]))
    filter0.append(X[0])

    print('  - get filtered data (filtered=%s)' % (filter0.filtered))
    assert(abs(filter0.filtered - X[0]) < epsilon)

    print('  - append data (x=%f)' % (X[1]))
    filter0.append(X[1])

    print('  - get filtered data (filtered=%s)' % (filter0.filtered))
    assert(abs(filter0.filtered - (X[0] * d + X[1]) / (d + 1)) < epsilon)

    print('  - append data (x=%f)' % (X[2]))
    filter0.append(X[2])

    print('  - get filtered data (filtered=%s)' % (filter0.filtered))
    assert(abs(filter0.filtered - (X[0] * d**2 + X[1] * d + X[2]) / (d**2 + d + 1)) < epsilon)

    print('\nTEST 2:')

    print('  - create decay filter (d=%f, wsize=%d)' % (d, wsize))
    filter1 = DecayFilter(d=d, wsize=wsize)

    X = [np.array([[1, 1, 1], [2, 2, 2]]),
         np.array([[1, 1, 1], [2, 2, 2]]),
         np.array([[10, 10, 10], [25, 5, -1000]])]

    print('  - append data (x=\n%s)' % (X[0]))
    filter1.append(X[0])

    print('  - get filtered data (filtered=\n%s)' % (filter1.filtered))
    assert(np.linalg.norm(filter1.filtered - X[0]) < epsilon)

    print('  - append data (x=\n%s)' % (X[1]))
    filter1.append(X[1])

    print('  - get filtered data (filtered=\n%s)' % (filter1.filtered))
    assert(np.linalg.norm(filter1.filtered - (X[0] * d + X[1]) / (d + 1)) < epsilon)

    print('  - append data (x=\n%s)' % (X[2]))
    filter1.append(X[2])

    print('  - get filtered data (filtered=\n%s)' % (filter1.filtered))
    assert(np.linalg.norm(filter1.filtered - (X[0] * d**2 + X[1] * d + X[2]) / (d**2 + d + 1)) < epsilon)

    print('\nRESULT: ALL TESTS PASSED!')


if __name__ == '__main__':
    unit_test()
