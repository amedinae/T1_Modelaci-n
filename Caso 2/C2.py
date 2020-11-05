# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 20:07:19 2020

@author: Andres Medina
"""



def dXdt(x,):
    return np.array([Q3*c3-Q4/V*x[0],
                     Q5c5+Q4/V*x[0]+QA/V*x[3],
                     QB/V*x{1}-2*QC/V*x[2],
                     Q6*c6+QC/V*x[2]-QA/V/x[4]
                     ])