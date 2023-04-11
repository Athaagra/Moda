#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 04:13:23 2023

@author: Optimus
"""
import numpy as np
pm= 0.9
pc=0.3
eta_m=2
eps=1e-7
parent_P=[0,1,0,1,1,1]
decision_variables=np.arange(0,len(parent_P))
gp=np.random.randint(0,10,1)/100
ymin=min(decision_variables)
ymax=max(decision_variables)
for j in decision_variables:
    if gp <= pm:
        y=parent_P[j]
        mut_pow=1.0/(eta_m+1.0)
        rn=np.random.randint(0,10,1)/100
        print(rn)
        if rn <= 0.5:
            val = 2.0* rn
            deltagen = val**mut_pow - 1.0
        else:
            val=1/(2.0*(1.0-rn))
            deltagen=1.0 - val**mut_pow        
        y=y+deltagen*(ymax-ymin)
        print(y)
        if y<ymin:
            y=ymin
        if y>ymax:
            y=ymax
        parent_P[j]=y