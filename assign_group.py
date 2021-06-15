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
#                                       Jim Mainprice on Tuesday 15th June 2021

# https://stackoverflow.com/questions/54226686/solving-assignment-problem-with-conditional-minimum-group-sizes-using-cvxpy
# TODO implement and test

import numpy as np
import cvxpy as cp

preference = np.array([[1, 2, 3],
                       [1, 2, 3],
                       [1, 2, 3],
                       [1, 2, 3],
                       [1, 2, 3],
                       [1, 3, 2]])
groupmax = np.array([3, 3, 3])

# Variables
selection = cp.Variable(shape=preference.shape, boolean=True)
bind_2 = cp.Variable(shape=preference.shape[1], boolean=True)
bind_3 = cp.Variable(shape=preference.shape[1], boolean=True)

# Constraints
group_constraint_1 = cp.sum(selection, axis=0) <= groupmax
group_constraint_2 = (1 - bind_2) * 2 >= 2 - cp.sum(selection, axis=0)
group_constraint_3 = (1 - bind_3) * 4 >= cp.sum(selection, axis=0)
bind_constraint = bind_2 + bind_3 == 1
assignment_constraint = cp.sum(selection, axis=1) == 1

# cost
cost = cp.sum(cp.multiply(preference, selection))

constraints = [group_constraint_1,
               group_constraint_2,
               group_constraint_3,
               bind_constraint,
               assignment_constraint]

# Create problem and solve
assign_prob = cp.Problem(cp.Minimize(cost), constraints)
assign_prob.solve(solver=cp.GLPK_MI)
print(selection.value)
