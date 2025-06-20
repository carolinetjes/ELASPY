#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import sys

import numpy.random as rnd


FRACTIONS = [0.035, 0.031, 0.026, 0.022, 0.018, 0.018, 0.018, 0.022, 0.031, 0.044, 0.057, 0.059, 0.062, 0.062, 0.059, 0.057, 0.055, 0.053, 0.053, 0.051, 0.048, 0.044, 0.04, 0.035] # Rounded using from "Total cases" from Table 1 of https://pubmed.ncbi.nlm.nih.gov/25664379/
LAMBDA_BY_HOUR = [x * 5 / 60 * 24 for x in FRACTIONS] # this is on average 5 calls per hour


def find_hour_index_and_remaining_minutes_in_this_hour(last_call_arrived):
    for hourIndex in range(len(LAMBDA_BY_HOUR)):
        if 60 * (hourIndex+1) > last_call_arrived:
            remainingMinutes = 60 * (hourIndex+1) - last_call_arrived
            return hourIndex, remainingMinutes
        
    sys.exit('Error: find_hour_index in time_varying_lambda.py failed.')
    return 100000, -1
    

def get_next_call_arrival_time(last_call_arrived, rng):
    """
    lambda varies by hour of the day (ignore day of week).
    Enter the last call arrival time (in minutes) and recieve the arrival time of the next call (in minutes)
    """
    result = last_call_arrived #init
    hourIndexOfLastCall, remainingMinutesInThisHr = find_hour_index_and_remaining_minutes_in_this_hour(last_call_arrived)
    houridx = hourIndexOfLastCall #init

    while houridx < len(LAMBDA_BY_HOUR):
        lambda_of_the_hour = LAMBDA_BY_HOUR[houridx]
        interarrival_time = rng.exponential(1 / lambda_of_the_hour)
        if interarrival_time <= remainingMinutesInThisHr:
            result = result + interarrival_time
            return result
        else: 
            result = result + remainingMinutesInThisHr
            remainingMinutesInThisHr = 60
            houridx = houridx + 1

    return 999999999999  #some high number, to indicate that next call lies outside of the 24-h planning period
    



#run this file to test using the following:
rng = rnd.default_rng(1)
print(f"next_call_arrival_time after 6 = {get_next_call_arrival_time(6, rng)}")
print(f"next_call_arrival_time after 61 = {get_next_call_arrival_time(61, rng)}")
print(f"next_call_arrival_time after 100 = {get_next_call_arrival_time(100, rng)}")
print(f"next_call_arrival_time after 200 = {get_next_call_arrival_time(200, rng)}")