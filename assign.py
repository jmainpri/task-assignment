#!/usr/bin/env python
# This code solves linear sum assignment problem, also
# known as minimum weight matching in bipartite graphs
# See the assignment problem
# https://en.wikipedia.org/wiki/Assignment_problem
# It is useful for assigning jobs to students depending
# on preferences. You can simply define the name and
# preferences in a yaml file.
# WARNING the indices start with one in the yaml file
# and 0 in the code.

import yaml
import numpy as np
from scipy.optimize import linear_sum_assignment

# file = "example.yaml",
file = "students.yaml"
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
# in the vector form this matrix will be converted
# to a row major format [row_0, row_1, ..., row_N]
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
