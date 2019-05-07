#!/usr/bin/env python

# Copyright (c) 2010, University of Stuttgart
# All rights reserved.
#
# Permission to use, copy, modify, and distribute this software for any purpose
# with or without   fee is hereby granted, provided   that the above  copyright
# notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS  SOFTWARE INCLUDING ALL  IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR  BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR  ANY DAMAGES WHATSOEVER RESULTING  FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
# OTHER TORTIOUS ACTION,   ARISING OUT OF OR IN    CONNECTION WITH THE USE   OR
# PERFORMANCE OF THIS SOFTWARE.
#
#                                        Jim Mainprice on Sunday May 7 2019

# This code solves the linear sum assignment problem, also known as minimum
# weight matching in bipartite graphs

# See the assignment problem:
# [Wikipedia](https://en.wikipedia.org/wiki/Assignment_problem)

# It is useful for assigning jobs to students depending
# on preferences. You can simply define the name and
# preferences in a yaml file.
# WARNING the indices start with one in the yaml file and 0 in the code.

import yaml
import numpy as np
from scipy.optimize import linear_sum_assignment

file = "example.yaml",
with open(file, 'r') as stream:
    try:
        settings = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)
        exit(0)

preferences = []
for item in settings:
    preferences.append(next(iter(item.values())))

N = len(settings)
nb_preferences = len(preferences[0])
v_max = nb_preferences + 1.
v_min = 1.
alpha = (v_max - v_min) / nb_preferences

# Convention is
# - w[0, N] -> user one
# - w[1, N] -> user two
w = v_max * np.ones((N, N))
for i, p_u in enumerate(preferences):
    w_k = 0.
    for p in p_u:
        w[i, p - 1] = w_k
        w_k += alpha

row_ind, col_ind = linear_sum_assignment(w)
for idx in row_ind:
    print("{:4} - ({:14}) \t-> {}".format(
        idx + 1,
        next(iter(settings[idx].keys())),
        col_ind[idx] + 1))
