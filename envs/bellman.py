# -*- coding: utf-8 -*-
"""
@File    : bellman.py
@Time    : 2025/1/11 10:07
@Author  : zhangjian
@Email   : your_email@example.com
@Desc    : 
"""
import sympy
from sympy import symbols
sympy.init_printing()
v_hungry, v_full = symbols("v_hungry v_full")
q_hungry_eat, a_hungry_none, q_full_eat, q_full_none = symbols('q_hungry_eat a_hungry_none q_full_eat q_full_none')
alpha, beta, x, y, gamma = symbols('alpha, beta, x, y, gamma')
system = sympy.Matrix((
    (1,0,x-1,-x,0,0,0),
    (0,1,0,0,-y,y-1,0),
    (-gamma,0,1,0,0,0,-2),
    ((alpha-1)*gamma,-alpha*gamma,0,1,0,0,4*alpha-3),
    (-beta*gamma, (beta-1)*gamma,0,0,1,0,-4*beta+2),
    (0,-gamma,0,0,0,1,1)))
sol = sympy.solve_linear_system(system,v_hungry,v_full,q_hungry_eat, a_hungry_none, q_full_eat, q_full_none)
print(sol)
