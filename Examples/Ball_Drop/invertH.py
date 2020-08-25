#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:18:07 2020

@author: granthutchings
"""
import numpy as np
from scipy import optimize
import random
# =============================================================================
# def invertHtrue(h, g, C, R, et):
#     """
#     Generate synthetic experimental data t(h) for ball drop example
#     truth d^2h / dt^2 = g - (C / R) (dh / dt)^2
#        --> h(t) = \ln(1 + exp{2t \sqrt{g C/R}) / (C/R) - t / \sqrt{(C/R) (1/g)}
#     
#     :param h: vector -- starting heights of the balls we're dropping
#     :param g: scalar -- gravity
#     :param C: scalar -- coefficient of drag (air resistance)
#     :param R: vector -- radius of the ball we're dropping
#     :param et: scalar -- standard deviation of the observed time t given a hight h
#     :return: t(h) -- vector of times
#     """
#     # We have h(t), but we want t(h) for the tower problem. We use an optimizer for the inversion.
#     # get dims for
#     t = np.zeros(np.size(h))
#     for i in range(np.size(t)):
#         banana = lambda t_i: np.power((h[i] - dragheight(t_i,g,C,R)),2) # function to minimize
#         
#         # starting point based on no drag
#         t0 = np.sqrt(2*h[i]/g)
#         
#         # find the time
#         t_i = optimize.fmin(banana, t0)
#         
#         # add observation error
#         t[i] = t_i + et*np.random.uniform()
#             
#     return(t)
# =============================================================================
def invertHtrue(h, g, C, R, et):
    """
    Generate synthetic experimental data t(h) for ball drop example
    truth d^2h / dt^2 = g - (C / R) (dh / dt)^2
       --> h(t) = \ln(1 + exp{2t \sqrt{g C/R}) / (C/R) - t / \sqrt{(C/R) (1/g)}
    
    :param h: vector -- starting heights of the balls we're dropping
    :param g: scalar -- gravity
    :param C: scalar -- coefficient of drag (air resistance)
    :param R: vector -- radii of the balls we're dropping
    :param et: scalar -- standard deviation of the observed time t given a hight h
    :return: t(h) -- vector of times
    """
    def dragHeightTrue(t, g, C, R):
        """
        Given a time t, what's the height from which the ball was dropped
        """
        h = R/C * np.log( np.cosh( np.sqrt((C/R) * g) * t ) )
        return(h)

    # We have h(t), but we want t(h) for the tower problem. We use an optimizer for the inversion.
    t = np.zeros(shape=(np.size(R),np.size(h)))
    for rr in range(np.size(R)):
        for hh in range(np.size(h)):
            banana = lambda t_tmp: np.power((h[hh] - dragHeightTrue(t_tmp,g,C,R[rr])),2) # function to minimize
            
            # starting point based on no drag
            t0 = np.sqrt(2*h[hh]/g)
            
            # find the time
            t_tmp = optimize.fmin(banana, t0, disp=False)
            
            # add observation error
            t[rr,hh] = t_tmp + et*np.random.uniform()
            
    return(t)


def invertHsim(h, g, C, R):
    """
    Generate synthetic simulated data t(h) for ball drop example
    simulation d^2h / dt^2 = g - (C / R) (dh / dt)
       --> h(t) = exp{-(C/R) * t} / (C/R)^2 + g * t / (C/R)
    
    :param h: vector -- starting heights of the balls we're dropping
    :param g: vector -- gravity values to try
    :param C: vector -- coefficients of drag to try (air resistance)
    :param R: vector -- radii of the balls we're dropping
    :return: t(h) -- vector of times
    """
    def dragHeightSim(t, g, C, R):
        """
        Given a time t, what's the height from which the ball was dropped
        """
        b = C / R
        bsq = b**2
        # K1 and K2 are constants of integration
        K1 = -np.log(g)/b
        K2 = -g/bsq
        h = g*t/b + np.exp(-b*(t+K1))/bsq + K2
        return(h)

    # We have h(t), but we want t(h) for the tower problem. We use an optimizer for the inversion.
    # Need to solve for t given h, and do this for each setting of g, C, and R.
    
    #t = np.zeros(shape=(np.size(R),np.size(h),np.size(C),np.size(g)))
    t = np.zeros(shape=(np.size(R),np.size(h)))
    for rc in range(np.size(R)):
        for hh in range(np.size(h)):
                #for gg in range(np.size(g)):
                    #banana = lambda t_tmp: np.power((h[hh] - dragHeightSim(t_tmp,g[gg],C[cc],R[rr])),2) # function to minimize
                    banana = lambda t_tmp: np.power((h[hh] - dragHeightSim(t_tmp,g,C[rc],R[rc])),2)
                    # starting point - no drag
                    #t0 = np.sqrt(2*h[hh]/g[gg])
                    t0 = np.sqrt(2*h[hh]/g)
                    
                    # find the time
                    t_tmp = optimize.fmin(banana,t0,disp=False)
                    
                    #t[rr,hh,cc,gg] = t_tmp
                    t[rc,hh] = t_tmp
    return(t)

