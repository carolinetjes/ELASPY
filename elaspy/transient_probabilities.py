#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import copy, sys
import simpy as sp
import numpy as np
import pandas as pd

from typing import Any
from ambulance import Ambulance
from patient import Patient
from collections import deque


def any_vehicles_left_at_base( #
    env: sp.core.Environment,
    base_idx: int,
    ambulances: list[Ambulance],
    SIMULATION_DATA: dict[str, Any]
) -> tuple[bool, int]:
    """
    returns True if there is at least one ambulance at the base with index base_idx.
    """
    postal_code_base = SIMULATION_DATA["NODES_BASE_LOCATIONS"]["Base Locations"].tolist()[base_idx]

    for j in range(len(ambulances)):
        if not (
            ambulances[j].helps_patient or ambulances[j].assigned_to_patient
        ):
            if ambulances[j].drives_to_base:
                raise Exception("You should only use transient probabilities while TELEPORT_TO_BASE is true. Error.")
            else:
                ambulance_location_ID = ambulances[j].current_location_ID
            if ambulance_location_ID == postal_code_base:
                return True
    return False

def measure_transient_probabilities(env,  SIMULATION_PARAMETERS, SIMULATION_DATA, ambulances):
    """
    If average these over multiple runs, you get probabilities
      
    Occurs when SAVE_TRANSIENT_PROBABILITIES is True

    
    Alternative, more elaborate idea was to measure if a call from location i would be served from base j

    Parameters
    ----------
    SIMULATION_PARAMETERS : dict[str, Any]
      The parameter ``SAVE_TRANSIENT_PROBABILITIES`` should be True.

    Raises
    ------
    Exception
        If probabilities below 0 or above 1 are detected.

    Returns
    -------
    transient_probabilities : 3-dimensional np.ndarray, containing zeroes and ones.
    The first index is the minute,
    the second index whether you have a code red on the left node, 
    the third index whether you have a code red on the right node.

    """
    if SIMULATION_PARAMETERS["PROCESS_TYPE"] != "Time":
        raise Exception(
            "If SAVE_TRANSIENT_PROBABILITIES is True then you need PROCESS_TYPE to be Time. Please change it."
        )
    
    result = np.zeros(
        (
            SIMULATION_PARAMETERS["PROCESS_TIME"]+1, 
            len(SIMULATION_DATA["NODES_BASE_LOCATIONS"]), 
        ),
    )
    minutes_between_observations = 1
    while env.now <= SIMULATION_PARAMETERS["PROCESS_TIME"]:
        for j in range(len(SIMULATION_DATA["NODES_BASE_LOCATIONS"])):           
            if any_vehicles_left_at_base(env, j, ambulances, SIMULATION_DATA): 
                result[int(env.now),j] = 1
            else:
                result[int(env.now),j] = 0
        
        yield env.timeout(minutes_between_observations)  
    return result


"""
def base_index_with_nearest_idle_ambulance( #loosely based on check_select_ambulance. currently not used but maybe later
    env: sp.core.Environment,
    ambulances: list[Ambulance],
    SIMULATION_DATA: dict[str, Any],
    patient_location_ID: int,
) -> tuple[bool, int]:
    nr_ambulances_available = 0
    assignable_locations = []

    for j in range(len(ambulances)):
        if not (
            ambulances[j].helps_patient or ambulances[j].assigned_to_patient
        ):
            nr_ambulances_available += 1
            if ambulances[j].drives_to_base:
                raise Exception("You should only use transient probabilities while TELEPORT_TO_BASE is true. Error.")
            else:
                ambulance_location_ID = ambulances[j].current_location_ID

            #todo make sure it's diesel
            assignable_locations.append(ambulance_location_ID)

    if len(assignable_locations) == 0:
        VEHICLE_FREE = False
        base_idx = -1

    else:
        VEHICLE_FREE = True
        postal_code_shortest_time = (
            SIMULATION_DATA["SIREN_DRIVING_MATRIX"]
            .loc[assignable_locations, patient_location_ID]
            .idxmin()
        )
        #get the base index belonging to postal_code_shortest_time:
        base_idx = SIMULATION_DATA["NODES_BASE_LOCATIONS"]["Base Locations"].tolist().index(postal_code_shortest_time)
        
    return VEHICLE_FREE, base_idx
"""