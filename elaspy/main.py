#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the main interface of the simulator.

You can set all simulator parameters with this script and then run the
simulator. Below, all parameters are discussed. The input data itself is
discussed in the input data section of the website.

Parameters
----------
START_SEED_VALUE : int
    The initial seed value. The seed of the ith run is equal to
    ``START_SEED_VALUE  + (i-1)``.
DATA_DIRECTORY : str
    The folder, relative to the ``ROOT_DIRECTORY`` (automatically determined),
    where the input data is located.
SIMULATION_INPUT_DIRECTORY : str | None
    The folder, relative to the ``ROOT_DIRECTORY`` (automatically determined),
    where the simulation input data is located if ``LOAD_INPUT_DATA=True``.
    Otherwise it should be ``None``.
SIMULATION_OUTPUT_DIRECTORY : str
    The folder, relative to the ``ROOT_DIRECTORY`` (automatically determined),
    where the simulation output data should be saved.
TRAVEL_TIMES_FILE : str
    The name of the file that contains the data with the siren travel times
    between nodes.
DISTANCE_FILE : str
    The name of the file that contains the data with the distance between nodes.
NODES_FILE : str
    The name of the file that contains the data with the nodes of the region.
HOSPITAL_FILE : str
    The name of the file that contains the nodes where hospitals are located.
BASE_LOCATIONS_FILE : str
    The name of the file that contains the nodes where bases are located.
AMBULANCE_BASE_LOCATIONS_FILE : str:
    The name of the file that contains the assignment of ambulances to bases.
SCENARIO : str
    The scenario. The following are valid: RB1, RB2, FB1, RB1_FB1, RB1_RH1,
    RB1_FH1, FB1_RH1, FB1_FH1, RB50_RH50, Diesel.
CHARGING_SCENARIO_FILE : str
    The name of the file that contains the charging scenario data.
SIMULATION_PATIENT_OUTPUT_FILE_NAME : str
    The name of the file where the patient dataframe will be saved.
SIMULATION_AMBULANCE_OUTPUT_FILE_NAME : str
    The name of the file where the ambulance dataframe will be saved.
RUN_PARAMETERS_FILE_NAME : str
    The name of the file where the run parameters will be saved.
RUNNING_TIME_FILE_NAME : str
    The name of the file where the running times will be saved.
SIMULATION_PRINTS_FILE_NAME : str
    The name of the file where the simulation prints will be saved.
MEAN_RESPONSE_TIMES_FILE_NAME : str
    The name of the file where the mean response time of each run will be saved.
EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME : str
    The name of the file where the 95% empirical quantile of the response time
    of each run will be saved.
BUSY_FRACTIONS_FILE_NAME : str
    The name of the file where the empirical busy fraction of each run will be
    saved.
INTERARRIVAL_TIMES_FILE : str
    The name of the file with the interarrival times of the patients if
    ``LOAD_INPUT_DATA=True``. Otherwise it should be ``None``.
ON_SITE_AID_TIMES_FILE : str
    The name of the file with the on-site aid times of the patients if
    ``LOAD_INPUT_DATA=True``. Otherwise it should be ``None``.
DROP_OFF_TIMES_FILE : str
    The name of the file with the handover times at the hospital of the
    patients if ``LOAD_INPUT_DATA=True``. Otherwise it should be ``None``.
LOCATION_IDS_FILE : str
    The name of the file with the patient arrival locations if
    ``LOAD_INPUT_DATA=True``. Otherwise it should be ``None``.
TO_HOSPITAL_FILE : str
    The name of the file that specifies for each patient whether transportation
    to the hospital is required or not if ``LOAD_INPUT_DATA=True``. Otherwise
    it should be ``None``.
NUM_RUNS : int
    The number of simulation runs.
PROCESS_TYPE : str
    The type of arrival process. Use "Time" to simulate an arrival process
    where patients arrive within ``PROCESS_TIME`` time. Use "Number" to
    simulate an arrival process where ``PROCESS_NUM_CALLS`` patients arrive.
PROCESS_NUM_CALLS : int | None
    The number of patients that arrive if ``PROCESS_TYPE="Number"``. Should be
    ``None`` if ``PROCESS_TYPE="Time"``.
PROCESS_TIME : float | None
    The time in minutes during which patients can arrive if
    ``PROCESS_TYPE="Time"``. Should be ``None`` if ``PROCESS_TYPE="Number"``.
NUM_AMBULANCES : int
    The number of ambulances.
PROB_GO_TO_HOSPITAL : float | None
    The probability that a patient has to be transported to a hospital if
    ``LOAD_INPUT_DATA=False``. Otherwise it should be ``None``.
CALL_LAMBDA : float | None
    The arrival rate parameter of the arrival Poisson process if
    ``LOAD_INPUT_DATA=False``. Otherwise it should be ``None``.
AID_PARAMETERS : list[float | int]
    The parameters of the lognormal distribution for providing treatment on
    site. The first parameter is the sigma parameter, the second the
    location parameter , the third the scale parameter and the last the
    cut-off/maximum value.
DROP_OFF_PARAMETERS : list[float | int]
    The parameters of the lognormal distribution for the handover time at the
    hospital. The first parameter is the sigma parameter, the second the
    location parameter , the third the scale parameter and the last the
    cut-off/maximum value.
ENGINE_TYPE : str
    The engine type. Either "electric" or "diesel".
IDLE_USAGE :  float | None
    The energy consumption when idle/stationary in kW. If
    ``ENGINE_TYPE="diesel"`` it should be ``None``.
DRIVING_USAGE :  float | None
    The energy consumption when driving in kWh/km. If ``ENGINE_TYPE="diesel"``
    it should be ``None``.
BATTERY_CAPACITY : float
    The battery capacity of an electric ambulance. If ``ENGINE_TYPE="diesel"``
    it is equal to infinity (i.e., ``numpy.inf``)).
NO_SIREN_PENALTY : float
    The penalty for driving without sirens. The driving times with siren are
    scaled according to this value. Should be between 0 and 1.
LOAD_INPUT_DATA : bool
    Whether the input data should be read from data (``True``) or generated
    before the simulation starts (``False``).
CRN_GENERATOR : str | None
    The pseudo-random number generator that should be used if
    `LOAD_INPUT_DATA=False``. Either "Generator" for using NumPy's default or
    "RandomState" for Numpy's legacy generator. It should be ``None`` if
    ``LOAD_INPUT_DATA=True``.
INTERVAL_CHECK_WP : float | None
    The interval (in minutes) at which the simulator checks for waiting
    patients. If ``ENGINE_TYPE="diesel"`` it should be ``None``.
TIME_AFTER_LAST_ARRIVAL : float | None
    The time after the last arriving patient the simulator needs to check for
    waiting patients. If ``ENGINE_TYPE="diesel"`` it should be ``None``.
AT_BOUNDARY : float
    The warm-up period (in minutes) for the busy fraction calculation.
FT_BOUNDARY : float
    The cool-down period (in minutes) for the busy fraction calculation.
TELEPORT_TO_BASE : bool
    If true, the ambulance, when becoming idle again, is immediately at its base, 
    in zero amount of travel time. If electric, this does not change the amount 
    of electricity to get there, as electricity is based on distance and not time.
PRINT : bool
    If ``True``, debug prints are provided that clarify the simulation process.
PRINT_STATISTICS : bool
    If ``True``, useful simulation statistics such as the mean response time
    are provided for each run.
PLOT_FIGURES : bool
    If ``True``, multiple plots are provided for each simulation run.
SAVE_PRINTS_TXT : bool
    If ``True``, the prints are saved to ``SIMULATION_PRINTS_FILE_NAME``.
SAVE_OUTPUT : bool
    If ``True``, saves the run parameters in ``RUN_PARAMETERS_FILE_NAME``, the
    running times in ``RUNNING_TIME_FILE_NAME``, the mean response time per run
    in ``MEAN_RESPONSE_TIMES_FILE_NAME``, the 95% empirical quantile of the
    response time per run in ``EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME`` and the
    empirical busy fraction in ``BUSY_FRACTIONS_FILE_NAME``.
SAVE_PLOTS : bool
    If ``True``, saves the plots. Can only be ``True`` if ``PLOT_FIGURES=True``.
SAVE_DFS : bool
    If ``True``, saves the ambulance dataframe of each run in
    ``SIMULATION_AMBULANCE_OUTPUT_FILE_NAME`` (adding a run_i suffix) and the
    patient dataframe of each run in ``SIMULATION_PATIENT_OUTPUT_FILE_NAME``
    (adding a run_i suffix).
DATA_COLUMNS_PATIENT : list[str]
    The columns for the patient DataFrame.
DATA_COLUMNS_AMBULANCE : list[str]
    The columns for the ambulance DataFrame.
"""
from typing import Any

import os
import sys
import copy
import scipy
from scipy import stats
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ambulance_simulation import run_simulation
from input_output_functions import (
    print_parameters,
    save_simulation_output,
    simulation_statistics,
    calculate_response_time_ecdf,
    check_input_parameters,
    save_input_parameters,
    calculate_busy_fraction,
)
from plot_functions import (
    plot_battery_levels,
    plot_response_times,
    hist_battery_increase_decrease,
)


###################################Seed########################################
START_SEED_VALUE: int | None = 110
################################Directories####################################
ROOT_DIRECTORY: str = os.path.dirname(os.path.dirname(__file__))
DATA_DIRECTORY: str = os.path.join(ROOT_DIRECTORY, "dataToyExample/")
SIMULATION_INPUT_DIRECTORY: str | None = None
SIMULATION_OUTPUT_DIRECTORY: str = os.path.join(ROOT_DIRECTORY, "results/")
#################################File names####################################
TRAVEL_TIMES_FILE: str = "siren_driving_matrix_2022.csv"
DISTANCE_FILE: str = "distance_matrix_2022.csv"
NODES_FILE: str = "nodes_Utrecht_2021.csv"
HOSPITAL_FILE: str = "Hospital_Postal_Codes_Utrecht_2021.csv"
BASE_LOCATIONS_FILE: str = "RAVU_base_locations_Utrecht_2021.csv"
AMBULANCE_BASE_LOCATIONS_FILE: str = (
    "Base_Locations.csv"
)
SCENARIO: str = "Diesel" # "FB1_FH1"
CHARGING_SCENARIO_FILE: str = f"charging_scenario_21_22_{SCENARIO}.csv"
SIMULATION_PATIENT_OUTPUT_FILE_NAME: str = f"Patient_df_{SCENARIO}"
SIMULATION_AMBULANCE_OUTPUT_FILE_NAME: str = f"Ambulance_df_{SCENARIO}"

RUN_PARAMETERS_FILE_NAME: str = f"run_parameters_{SCENARIO}"
RUNNING_TIME_FILE_NAME: str = f"running_times_{SCENARIO}"
SIMULATION_PRINTS_FILE_NAME: str = f"run_prints_{SCENARIO}"
MEAN_RESPONSE_TIMES_FILE_NAME: str = f"mean_response_times_all_runs_{SCENARIO}"
EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME: str = (
    f"emp_quantile_response_times_all_runs_{SCENARIO}"
)
BUSY_FRACTIONS_FILE_NAME: str = f"busy_fractions_all_runs_{SCENARIO}"

INTERARRIVAL_TIMES_FILE: str | None = None
ON_SITE_AID_TIMES_FILE: str | None = None
DROP_OFF_TIMES_FILE: str | None = None
LOCATION_IDS_FILE: str | None = None
TO_HOSPITAL_FILE: str | None = None
############################Simulation parameters##############################
NUM_RUNS: int = 3000
PROCESS_TYPE: str = "Time"
PROCESS_NUM_CALLS: int | None = None
PROCESS_TIME: float | None =  100 #end of simulation horizon agreed with Ton. Nanne used 720 (720 mins = 12 hours)
NUM_AMBULANCES: int = 10
PROB_GO_TO_HOSPITAL: float | None = 0.6300
CALL_LAMBDA: float | None = 5/60   #4/60   #1/7.75
AID_PARAMETERS: list[float | int] = [0.38, -10.01, 37.00, 88]
DROP_OFF_PARAMETERS: list[float | int] | None = [0.39, -8.25, 35.89, 88]
ENGINE_TYPE: str = "diesel"
IDLE_USAGE: float | None = None # 5  # kW for electric
DRIVING_USAGE: float | None = None # choose  0.4  # kWh/km for electric
BATTERY_CAPACITY: float
if ENGINE_TYPE == "electric":
    BATTERY_CAPACITY = 150.0
else:
    BATTERY_CAPACITY = np.inf
NO_SIREN_PENALTY: float = 0.95
LOAD_INPUT_DATA: bool = False
CRN_GENERATOR: str | None = "Generator"
INTERVAL_CHECK_WP: float | None = None #1
TIME_AFTER_LAST_ARRIVAL: float | None = None #100
AT_BOUNDARY: float = 60.0
FT_BOUNDARY: float = 720.0
TELEPORT_TO_BASE: bool = True
##############################Output Parameters################################
PRINT: bool = False
PRINT_STATISTICS: bool = False
PLOT_FIGURES: bool = False

SAVE_PRINTS_TXT: bool = False
SAVE_OUTPUT: bool = False 
SAVE_PLOTS: bool = False
SAVE_DFS: bool = False

#for EMSPLEX:
SAVE_TRANSIENT_PROBABILITIES: bool = True # use in combination with PROCESS_TYPE = Time, so you know in advance how big of an array to initialize

DATA_COLUMNS_PATIENT: list["str"] = [
    "patient_ID",
    "response_time",
    "arrival_time",
    "location_ID",
    "nr_ambulances_available",
    "nr_ambulances_not_assignable",
    "assigned_to_ambulance_nr",
    "waiting_time_before_assigned",
    "driving_time_to_patient",
    "ambulance_arrival_time",
    "on_site_aid_time",
    "to_hospital",
    "hospital_ID",
    "driving_time_to_hospital",
    "drop_off_time_hospital",
    "finish_time",
]
DATA_COLUMNS_AMBULANCE: list["str"] = [
    "ambulance_ID",
    "time",
    "battery_level_before",
    "battery_level_after",
    "use_or_charge",
    "idle_or_driving_decrease",
    "idle_time",
    "source_location_ID",
    "target_location_ID",
    "driven_km",
    "battery_decrease",
    "charging_type",
    "charging_location_ID",
    "speed_charger",
    "charging_success",
    "waiting_time",
    "charging_interrupted",
    "charging_time",
    "battery_increase",
]
###################################Initialization##############################
SIMULATION_PARAMETERS: dict[str, Any] = {
    "START_SEED_VALUE": START_SEED_VALUE,
    "NUM_RUNS": NUM_RUNS,
    "PROCESS_TYPE": PROCESS_TYPE,
    "PROCESS_NUM_CALLS": PROCESS_NUM_CALLS,
    "PROCESS_TIME": PROCESS_TIME,
    "NUM_AMBULANCES": NUM_AMBULANCES,
    "PROB_GO_TO_HOSPITAL": PROB_GO_TO_HOSPITAL,
    "CALL_LAMBDA": CALL_LAMBDA,
    "AID_PARAMETERS": AID_PARAMETERS,
    "DROP_OFF_PARAMETERS": DROP_OFF_PARAMETERS,
    "ENGINE_TYPE": ENGINE_TYPE,
    "IDLE_USAGE": IDLE_USAGE,
    "DRIVING_USAGE": DRIVING_USAGE,
    "BATTERY_CAPACITY": BATTERY_CAPACITY,
    "NO_SIREN_PENALTY": NO_SIREN_PENALTY,
    "PRINT": PRINT,
    "DATA_COLUMNS_PATIENT": DATA_COLUMNS_PATIENT,
    "DATA_COLUMNS_AMBULANCE": DATA_COLUMNS_AMBULANCE,
    "TRAVEL_TIMES_FILE": TRAVEL_TIMES_FILE,
    "DISTANCE_FILE": DISTANCE_FILE,
    "NODES_FILE": NODES_FILE,
    "HOSPITAL_FILE": HOSPITAL_FILE,
    "BASE_LOCATIONS_FILE": BASE_LOCATIONS_FILE,
    "AMBULANCE_BASE_LOCATIONS_FILE": AMBULANCE_BASE_LOCATIONS_FILE,
    "SCENARIO": SCENARIO,
    "CHARGING_SCENARIO_FILE": CHARGING_SCENARIO_FILE,
    "SIMULATION_PATIENT_OUTPUT_FILE_NAME": SIMULATION_PATIENT_OUTPUT_FILE_NAME,
    "SIMULATION_AMBULANCE_OUTPUT_FILE_NAME": SIMULATION_AMBULANCE_OUTPUT_FILE_NAME,
    "SIMULATION_OUTPUT_DIRECTORY": SIMULATION_OUTPUT_DIRECTORY,
    "DATA_DIRECTORY": DATA_DIRECTORY,
    "SIMULATION_INPUT_DIRECTORY": SIMULATION_INPUT_DIRECTORY,
    "SAVE_OUTPUT": SAVE_OUTPUT,
    "LOAD_INPUT_DATA": LOAD_INPUT_DATA,
    "INTERARRIVAL_TIMES_FILE": INTERARRIVAL_TIMES_FILE,
    "ON_SITE_AID_TIMES_FILE": ON_SITE_AID_TIMES_FILE,
    "DROP_OFF_TIMES_FILE": DROP_OFF_TIMES_FILE,
    "LOCATION_IDS_FILE": LOCATION_IDS_FILE,
    "TO_HOSPITAL_FILE": TO_HOSPITAL_FILE,
    "CRN_GENERATOR": CRN_GENERATOR,
    "INTERVAL_CHECK_WP": INTERVAL_CHECK_WP,
    "TIME_AFTER_LAST_ARRIVAL": TIME_AFTER_LAST_ARRIVAL,
    "RUN_PARAMETERS_FILE_NAME": RUN_PARAMETERS_FILE_NAME,
    "RUNNING_TIME_FILE_NAME": RUNNING_TIME_FILE_NAME,
    "PLOT_FIGURES": PLOT_FIGURES,
    "SAVE_PLOTS": SAVE_PLOTS,
    "SAVE_DFS": SAVE_DFS,
    "SAVE_TRANSIENT_PROBABILITIES": SAVE_TRANSIENT_PROBABILITIES,
    "PRINT_STATISTICS": PRINT_STATISTICS,
    "SIMULATION_PRINTS_FILE_NAME": SIMULATION_PRINTS_FILE_NAME,
    "SAVE_PRINTS_TXT": SAVE_PRINTS_TXT,
    "MEAN_RESPONSE_TIMES_FILE_NAME": MEAN_RESPONSE_TIMES_FILE_NAME,
    "EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME": EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME,
    "AT_BOUNDARY": AT_BOUNDARY,
    "FT_BOUNDARY": FT_BOUNDARY,
    "TELEPORT_TO_BASE": TELEPORT_TO_BASE,
    "BUSY_FRACTIONS_FILE_NAME": BUSY_FRACTIONS_FILE_NAME,
}
SIMULATION_DATA: dict[str, Any] = {
    "DATA_COLUMNS_PATIENT": DATA_COLUMNS_PATIENT,
    "DATA_COLUMNS_AMBULANCE": DATA_COLUMNS_AMBULANCE,
}

####################################Run########################################
if __name__ == "__main__":
    start_time_script = datetime.datetime.now()

    if "Toy" in DATA_DIRECTORY:
        if NUM_AMBULANCES != 10 or CALL_LAMBDA != 5/60 or PROCESS_TIME != 100:
            raise Exception(
                "I think you want to run the toy example, but ambu, lambda, horizon = {NUM_AMBULANCES},{CALL_LAMBDA},{PROCESS_TIME}  while we agreed it should be 10, 1/12, 100 mins."
            )

    copy_simulation_parameters = copy.deepcopy(SIMULATION_PARAMETERS)
    check_input_parameters(SIMULATION_PARAMETERS)

    if SIMULATION_PARAMETERS["SAVE_OUTPUT"]:
        save_input_parameters(SIMULATION_PARAMETERS)
    else:
        print_parameters(SIMULATION_PARAMETERS)

    if SIMULATION_PARAMETERS["SAVE_PRINTS_TXT"]:
        sys.stdout = open(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['SIMULATION_PRINTS_FILE_NAME']}.txt",
            "wt",
        )


    mean_response_times: np.ndarray = np.zeros((NUM_RUNS))
    emp_quantile_response_times: np.ndarray = np.zeros((NUM_RUNS))
    busy_fractions: np.ndarray = np.zeros(NUM_RUNS)
    running_times: np.ndarray = np.zeros(NUM_RUNS)
    transient_probabilities = []

    for run_nr in range(NUM_RUNS):
        print(f"Run nr: {run_nr}.")

        if not SIMULATION_PARAMETERS["LOAD_INPUT_DATA"]:
            SIMULATION_PARAMETERS["SEED_VALUE"] = (
                SIMULATION_PARAMETERS["START_SEED_VALUE"] + run_nr
            )

        start_time_simulation_run = datetime.datetime.now()
        run_simulation(SIMULATION_PARAMETERS, SIMULATION_DATA)
        end_time_simulation_run = datetime.datetime.now()
        running_times[run_nr] = (
            end_time_simulation_run - start_time_simulation_run
        ).total_seconds()

        # Create DataFrames of simulation output
        start_time_df = datetime.datetime.now()
        df_patient = pd.DataFrame(
            SIMULATION_DATA["output_patient"],
            columns=SIMULATION_PARAMETERS["DATA_COLUMNS_PATIENT"],
        )
        df_patient = calculate_response_time_ecdf(df_patient)
        df_ambulance = pd.DataFrame(
            SIMULATION_DATA["output_ambulance"],
            columns=SIMULATION_PARAMETERS["DATA_COLUMNS_AMBULANCE"],
        )
        print(
            "The running time for creating the dfs is: "
            f"{datetime.datetime.now()-start_time_df}."
        )

        #store the transient observations of 1(base has availability) or 0 (base empty):
        if SIMULATION_PARAMETERS["SAVE_TRANSIENT_PROBABILITIES"]:
            transient_probabilities.append(SIMULATION_DATA["transient_probabilities"]) 

        # Plot simulation output
        start_time_plots_stats = datetime.datetime.now()
        if SIMULATION_PARAMETERS["PLOT_FIGURES"]:
            plot_response_times(df_patient, run_nr, SIMULATION_PARAMETERS)
            if SIMULATION_PARAMETERS["ENGINE_TYPE"] == "electric":
                plot_battery_levels(
                    df_ambulance, run_nr, SIMULATION_PARAMETERS
                )
                hist_battery_increase_decrease(
                    df_ambulance, run_nr, SIMULATION_PARAMETERS
                )
        if SIMULATION_PARAMETERS["PRINT_STATISTICS"]:
            simulation_statistics(
                df_patient,
                df_ambulance,
                start_time_simulation_run,
                end_time_simulation_run,
                SIMULATION_DATA["nr_times_no_fast_no_regular_available"],
                SIMULATION_PARAMETERS,
            )
        print(
            "The running time for creating the plots and printing the "
            "simulation stats is: "
            f"{datetime.datetime.now()-start_time_plots_stats}."
        )

        # Save simulation output
        if SIMULATION_PARAMETERS["SAVE_DFS"]:
            start_time_saving = datetime.datetime.now()
            save_simulation_output(
                SIMULATION_PARAMETERS["SIMULATION_OUTPUT_DIRECTORY"],
                SIMULATION_PARAMETERS["SIMULATION_PATIENT_OUTPUT_FILE_NAME"],
                df_patient,
                run_nr,
            )
            save_simulation_output(
                SIMULATION_PARAMETERS["SIMULATION_OUTPUT_DIRECTORY"],
                SIMULATION_PARAMETERS["SIMULATION_AMBULANCE_OUTPUT_FILE_NAME"],
                df_ambulance,
                run_nr,
            )
            print(
                "The running time for saving the data is: "
                f"{datetime.datetime.now()-start_time_saving}."
            )

        mean_response_times[run_nr] = np.mean(df_patient["response_time"])
        emp_quantile_response_times[run_nr] = np.min(
            df_patient.loc[df_patient["ecdf_rt"] >= 0.95]["response_time"]
        )
        busy_fractions[run_nr] = calculate_busy_fraction(
            df_patient, SIMULATION_PARAMETERS
        )

    m_mean_response_times = np.mean(mean_response_times)
    m_emp_quantile_response_times = np.mean(emp_quantile_response_times)
    m_busy_fractions = np.mean(busy_fractions)

    print("\nAll runs finished")
    print(
        "The mean mean response time over "
        f"all runs is: {m_mean_response_times}."
    )
    if NUM_RUNS > 1:
        if SIMULATION_PARAMETERS["SAVE_TRANSIENT_PROBABILITIES"]:
            transient_probabilities = np.stack(transient_probabilities) 
            mean_across_runs = np.mean(transient_probabilities, axis=0)
            np.set_printoptions(threshold=np.inf)  # disables truncation
            print(f"Estimates of transient probabilities: \n{mean_across_runs}")
            std_err = stats.sem(transient_probabilities, axis=0)
            confidence = 0.95  # 95% confidence intervals using t-distribution
            df = transient_probabilities.shape[0] - 1  # degrees of freedom 
            t_crit = stats.t.ppf((1 + confidence) / 2., df)  # close to 1.96 for large df
            
            margin_of_error = t_crit * std_err
            lower = mean_across_runs - margin_of_error
            upper = mean_across_runs + margin_of_error
            time = np.arange(transient_probabilities.shape[1])
            plt.figure(figsize=(10, 5))
            for b in range(2):
                plt.plot(time, mean_across_runs[:, b], label=f"Base {b} DES")
                plt.fill_between(time, lower[:, b], upper[:, b], alpha=0.3)#, label=f"Base {b} 95% CI")
            EmsplexWithLambda4 = [[1.0, 1.0],
[1.0, 1.0],
[1.0, 0.9998222696992047],
[1.0, 0.999471765100431],
[1.0, 0.9989535858458936],
[1.0, 0.9982730590523198],
[1.0, 0.9974358093862733],
[1.0, 0.996447825315387],
[0.9999999999356861, 0.9953155119045035],
[0.9999999994522252, 0.9940457225734091],
[0.9999999974120681, 0.9926457652904541],
[0.9999999910489543, 0.9911233820073924],
[0.9999999747146179, 0.9894867031563237],
[0.9999999382107696, 0.98774418135589],
[0.9999998647494213, 0.9859045099475704],
[0.999999728632002, 0.9839765326085058],
[0.9999994927755822, 0.981969150181363],
[0.9999991062384923, 0.9798912302004713],
[0.9999985019059822, 0.9777515235704248],
[0.9999975944900722, 0.9755585916500741],
[0.9999962789791361, 0.973320745763372],
[0.9999944296458476, 0.9710460000134703],
[0.9999918996909601, 0.9687420372941452],
[0.9999885215685267, 0.9664161876148321],
[0.9999841080082091, 0.9640754172956333],
[0.9999784537238255, 0.9617263272386092],
[0.9999713377747845, 0.9593751583189392],
[0.9999625265283911, 0.9570278019331436],
[0.9999517771555717, 0.954689813856911],
[0.99993884157966, 0.9523664297677117],
[0.999923470786887, 0.9500625810454535],
[0.9999054193977982, 0.9477829097502993],
[0.9998844503909337, 0.9455317819676815],
[0.9998603398640115, 0.9433132989889099],
[0.9998328817139924, 0.9411313060488006],
[0.999801892116355, 0.938989398561071],
[0.9997672136861805, 0.9368909259731946],
[0.9997287192096681, 0.9348389935033818],
[0.9996863148446479, 0.9328364621241562],
[0.9996399427024776, 0.9308859472221743],
[0.999589582741036, 0.9289898163963017],
[0.9995352539187571, 0.9271501868600309],
[0.9994770145819305, 0.9253689228950608],
[0.9994149620808517, 0.9236476337652098],
[0.999349231633743, 0.9219876724486639],
[0.9992799944796298, 0.9203901354863101],
[0.9992074553815501, 0.9188558641786311],
[0.9991318495587489, 0.9173854472967949],
[0.9990534391402204, 0.9159792254081004],
[0.9989725092416833, 0.914637296854162],
[0.9988893637736224, 0.9133595253639322],
[0.9988043210894448, 0.9121455492341223],
[0.9987177095803408, 0.9109947919675914],
[0.9986298633175139, 0.90990647422621],
[0.9985411178336258, 0.908879626928605],
[0.9984518061241956, 0.9079131053047946],
[0.9983622549370037, 0.9070056037085483],
[0.9982727814039122, 0.9061556709837147],
[0.998183690055584, 0.9053617261819791],
[0.9980952702459034, 0.9046220744357123],
[0.99800779399995, 0.9039349227999188],
[0.9979215142875438, 0.9032983958908947],
[0.9978366637139185, 0.9027105511653029],
[0.9977534536101842, 0.902169393701154],
[0.9976720734989938, 0.9016728903610015],
[0.9975926909052261, 0.9012189832368674],
[0.9975154514774983, 0.9008056022955284],
[0.9974404793837949, 0.9004306771613322],
[0.9973678779433123, 0.9000921479913538],
[0.9972977304565872, 0.8997879754141493],
[0.9972301011969182, 0.8995161495184337],
[0.9971650365278234, 0.899274697891552],
[0.9971025661136065, 0.8990616927195648],
[0.997042704192876, 0.8988752569711028],
[0.9969854508879117, 0.8987135696958737],
[0.9969307935259701, 0.8985748704758983],
[0.9968787079518577, 0.8984574630732648],
[0.9968291598142687, 0.8983597183225508],
[0.9967821058114301, 0.8982800763191625],
[0.9967374948844511, 0.8982170479568208],
[0.9966952693493957, 0.8981692158683965],
[0.9966553659614802, 0.898135234824409],
[0.9966177169069053, 0.8981138316428559],
[0.9965822507196817, 0.898103804662778],
[0.9965488931223916, 0.8981040228321858],
[0.9965175677911698, 0.8981134244587825],
[0.9964881970462872, 0.8981310156694142],
[0.9964607024706134, 0.8981558686214568],
[0.9964350054589298, 0.8981871195064535],
[0.9964110277015886, 0.8982239663833579],
[0.9963886077250746, 0.8982649384499407],
[0.9963676775042879, 0.8983094173391188],
[0.9963481692970343, 0.8983568332179351],
[0.9963300159836674, 0.8984066620183321],
[0.9963131513419907, 0.8984584226814428],
[0.9962975102577252, 0.8985116743569569],
[0.996283028886044, 0.8985660136360747],
[0.9962696447781538, 0.8986210718903545],
[0.9962572969845487, 0.8986765127746977],
[0.9962459261437699, 0.8987320299342881],
[0.9962459261437699, 0.8987320299342881]
]
            EmsplexHalfDetailed = [
[1.0, 1.0],
[1.0, 1.0],
[1.0, 1.0],
[1.0, 1.0],
[1.0, 0.9988895276845842],
[1.0, 0.9988895276845842],
[1.0, 0.9967486238909735],
[1.0, 0.9967486238909735],
[1.0, 0.9936636472933029],
[1.0, 0.9936636472933029],
[1.0, 0.9897298117003562],
[1.0, 0.9897298117003562],
[1.0, 0.9850520471144194],
[1.0, 0.9850520471144194],
[1.0, 0.9797434902700052],
[1.0, 0.9797434902700052],
[0.9999999101936814, 0.9739219152732369],
[0.9999999101936814, 0.9739219152732369],
[0.9999993161074229, 0.9677051333009931],
[0.9999993161074229, 0.9677051333009931],
[0.9999971255945848, 0.9612065073599905],
[0.9999971255945848, 0.9612065073599905],
[0.9999911922260479, 0.9545314203946964],
[0.9999911922260479, 0.9545314203946964],
[0.9999780300385814, 0.9477750742928734],
[0.9999780300385814, 0.9477750742928734],
[0.9999527097632405, 0.9410215898687262],
[0.9999527097632405, 0.9410215898687262],
[0.9999089788247792, 0.9343441206328225],
[0.9999089788247792, 0.9343441206328225],
[0.9998395993231611, 0.9278055950230983],
[0.9998395993231611, 0.9278055950230983],
[0.9997368621583357, 0.9214597239526855],
[0.9997368621583357, 0.9214597239526855],
[0.9995932135148058, 0.9153520004775022],
[0.9995932135148058, 0.9153520004775022],
[0.9994019194557929, 0.9095205300738611],
[0.9994019194557929, 0.9095205300738611],
[0.9991576927597724, 0.9039966320368317],
[0.9991576927597724, 0.9039966320368317],
[0.9988572123281796, 0.8988052284622138],
[0.9988572123281796, 0.8988052284622138],
[0.9984994792713268, 0.8939650825180042],
[0.9984994792713268, 0.8939650825180042],
[0.9980859742587437, 0.8894889651792006],
[0.9980859742587437, 0.8894889651792006],
[0.9976206055160352, 0.8853838260666653],
[0.9976206055160352, 0.8853838260666653],
[0.9971094622450059, 0.8816510273442373],
[0.9971094622450059, 0.8816510273442373],
[0.99656041012002, 0.878286676955695],
[0.99656041012002, 0.878286676955695],
[0.9959825805014015, 0.8752820744049957],
[0.9959825805014015, 0.8752820744049957],
[0.9953858113311971, 0.8726242625090616],
[0.9953858113311971, 0.8726242625090616],
[0.9947800954353939, 0.8702966640408571],
[0.9947800954353939, 0.8702966640408571],
[0.9941750828333499, 0.8682797734795871],
[0.9941750828333499, 0.8682797734795871],
[0.9935796702716957, 0.8665518707533184],
[0.9935796702716957, 0.8665518707533184],
[0.9930016963777116, 0.86508972487476],
[0.9930016963777116, 0.86508972487476],
[0.9924477469809821, 0.8638692594695794],
[0.9924477469809821, 0.8638692594695794],
[0.9919230639361901, 0.8628661581225163],
[0.9919230639361901, 0.8628661581225163],
[0.9914315429971988, 0.8620563941290371],
[0.9914315429971988, 0.8620563941290371],
[0.9909758020049072, 0.8614166757933196],
[0.9909758020049072, 0.8614166757933196],
[0.9905572993985822, 0.8609248042750713],
[0.9905572993985822, 0.8609248042750713],
[0.9901764841033235, 0.8605599458202118],
[0.9901764841033235, 0.8605599458202118],
[0.9898329603751682, 0.8603028238774022],
[0.9898329603751682, 0.8603028238774022],
[0.989525654465188, 0.8601358391177589],
[0.989525654465188, 0.8601358391177589],
[0.9892529734140307, 0.8600431268526638],
[0.9892529734140307, 0.8600431268526638],
[0.9890129495069603, 0.8600105619542752],
[0.9890129495069603, 0.8600105619542752],
[0.9888033666714602, 0.8600257213164548],
[0.9888033666714602, 0.8600257213164548],
[0.988621867284409, 0.8600778133382593],
[0.988621867284409, 0.8600778133382593],
[0.9884660394693068, 0.8601575830366031],
[0.9884660394693068, 0.8601575830366031],
[0.9883334860615106, 0.8602572003402069],
[0.9883334860615106, 0.8602572003402069],
[0.9882208660002091, 0.8603662013278056],
[0.9882208660002091, 0.8603662013278056],
[0.9881261542844894, 0.8604799238233655],
[0.9881261542844894, 0.8604799238233655],
[0.9880474021624975, 0.8605946111712428],
[0.9880474021624975, 0.8605946111712428],
[0.9879827593829947, 0.8607072758150236],
[0.9879827593829947, 0.8607072758150236],
[0.9879827593829947, 0.8607072758150236]]
            
            Emsplex10timesMoreDetailed = [[1.0, 1.0],
[0.9999999999999998, 0.9998761538783657],
[0.9999999999995431, 0.9994831087742159],
[0.9999999999800162, 0.9988305201407913],
[0.999999999752897, 0.9979283340677122],
[0.9999999983776451, 0.9967868754900302],
[0.9999999927346785, 0.9954169369827618],
[0.9999999748369481, 0.9938298530366081],
[0.9999999275600884, 0.9920375466430013],
[0.9999998186410992, 0.990052539333562],
[0.9999995930108321, 0.9878879210181837],
[0.999999163753686, 0.9855572809883254],
[0.9999984021946914, 0.9830746055566898],
[0.9999971277691245, 0.9804541506042924],
[0.9999950984118042, 0.9777102987183304],
[0.9999920022040463, 0.974857410764179],
[0.9999874509413041, 0.9719096809042902],
[0.9999809761503112, 0.9688810025679431],
[0.9999720279142253, 0.9657848509948294],
[0.999959976682429, 0.962634185986986],
[0.9999441180697068, 0.9594413766128518],
[0.9999236805028191, 0.956218147955743],
[0.9998978354590887, 0.9529755486686958],
[0.9998657099628065, 0.9497239371204027],
[0.9998264009572151, 0.9464729832863278],
[0.9997789911457176, 0.9432316832218842],
[0.9997225658880406, 0.9400083829015681],
[0.9996562307383654, 0.9368108083624896],
[0.9995791292177696, 0.9336460993955785],
[0.9994904604198174, 0.9305208444293414],
[0.9993894960550925, 0.9274411147029601],
[0.9992755965490301, 0.9244124962893085],
[0.9991482258197678, 0.9214401189745327],
[0.9990069643816327, 0.9185286814078417],
[0.99885152044778, 0.9156824722890627],
[0.9986817387442282, 0.9129053876548334],
[0.9984976067978508, 0.9102009445544946],
[0.9982992585223527, 0.9075722915753062],
[0.9980869749972764, 0.9050222167878524],
[0.9978611824129867, 0.9025531537427903],
[0.9976224472359573, 0.9001671861666933],
[0.9973714687296431, 0.8978660519856027],
[0.9971090690428065, 0.8956511472578859],
[0.9968361811456142, 0.8935235305305078],
[0.9965538349509362, 0.8914839280517928],
[0.996263142001577, 0.8895327401851341],
[0.9959652791320865, 0.8876700492770654],
[0.9956614715257088, 0.8858956291438954],
[0.9953529755832199, 0.8842089562570407],
[0.9950410620020353, 0.8826092226307272],
[0.994726999432838, 0.8810953503486697],
[0.9944120390394235, 0.8796660076096163],
[0.994097400238136, 0.878319626125839],
[0.9937842578389197, 0.8770544196736162],
[0.9934737307533662, 0.8758684035701967],
[0.9931668723786865, 0.874759414836817],
[0.9928646627124849, 0.8737251328012688],
[0.992568002203349, 0.872763099895099],
[0.9922777072979663, 0.8718707424087251],
[0.991994507607662, 0.8710453909812614],
[0.9917190445864412, 0.8702843006197125],
[0.9914518715889364, 0.8695846700631249],
[0.9911934551599317, 0.8689436603303387],
[0.9909441773969049, 0.868358412314373],
[0.9907043392226783, 0.8678260633110694],
[0.9904741644060144, 0.86734376239421],
[0.9902538041730227, 0.8669086845727982],
[0.9900433422607022, 0.8665180436885633],
[0.9898428002749888, 0.86616910403237],
[0.9896521432285543, 0.8658591906769895],
[0.9894712851475995, 0.8655856985403534],
[0.989300094651392, 0.8653461002079892],
[0.989138400422808, 0.8651379525557227],
[0.988985996502231, 0.8649589022240233],
[0.9888426473505161, 0.8648066900036271],
[0.9887080926391105, 0.8646791541984018],
[0.9885820517366718, 0.8645742330360319],
[0.9884642278715585, 0.864489966200044],
[0.988354311958334, 0.8644244955582885],
[0.988251986083955, 0.8643760651632418],
[0.988156926655634, 0.8643430205987599],
[0.9880688072175606, 0.8643238077461974],
[0.9879873009478103, 0.8643169710403822],
[0.9879120828499778, 0.8643211512828632],
[0.9878428316564434, 0.8643350830764132],
[0.9877792314618138, 0.8643575919408578],
[0.9877209731060944, 0.8643875911662946],
[0.9876677553276305, 0.8644240784555228],
[0.9876192857058991, 0.8644661324032946],
[0.9875751468600832, 0.8645124376142903],
[0.9875349262012489, 0.8645617098826418],
[0.9874983959830431, 0.8646133404740305],
[0.9874653339031636, 0.8646667776682521],
[0.9874355239226775, 0.8647215227366454],
[0.9874087568041265, 0.8647771258826095],
[0.9873848304406703, 0.8648331822448359],
[0.987363550039774, 0.8648893280629085],
[0.9873447282135617, 0.8649452370902405],
[0.9873281850155908, 0.865000617316184],
[0.9873137479517119, 0.8650552080325358],
[0.9873137479517119, 0.8650552080325358]]


            EmsplexAvailableProbabilities = [[1.0, 1.0],
[1.0, 1.0],
[1.0, 0.9997222964050073],
[1.0, 0.9991765375258366],
[1.0, 0.9983726063512856],
[1.0, 0.9973207327168835],
[1.0, 0.9960315831127288],
[1.0, 0.994516343833522],
[0.9999999996161546, 0.9927867826933848],
[0.9999999967699574, 0.990855277982236],
[0.9999999849217707, 0.9887348082834253],
[0.9999999484652317, 0.9864389020314998],
[0.9999998561319273, 0.9839815503483857],
[0.9999996525301947, 0.981377090194201],
[0.9999992482114816, 0.9786400669819147],
[0.9999985088811876, 0.9757850865731129],
[0.9999972445184317, 0.9728266662061135],
[0.9999951992262036, 0.9697790927040589],
[0.9999920426012311, 0.9666562945845062],
[0.9999873633062089, 0.9634717327318],
[0.9999806653687033, 0.9602383123349509],
[0.9999713675467167, 0.9569683170096566],
[0.9999588059140517, 0.9536733645240631],
[0.999942239647302, 0.9503643823894531],
[0.9999208598516205, 0.9470516007708413],
[0.9998938011488643, 0.943744559697792],
[0.999860155668418, 0.9404521273708765],
[0.9998189890237471, 0.9371825264108591],
[0.9997693578208401, 0.933943365128819],
[0.9997103282226432, 0.9307416712505209],
[0.999640995082032, 0.9275839259574059],
[0.9995605011521946, 0.9244760965669991],
[0.9994680558865936, 0.9214236666331824],
[0.9993629533514374, 0.9184316626759869],
[0.999244588793166, 0.915504677133415],
[0.99911247343345, 0.9126468874533963],
[0.9989662471059755, 0.909862071507068],
[0.9988056884034413, 0.907153619704594],
[0.9986307220693978, 0.9045245443345276],
[0.9984414234463561, 0.9019774867326352],
[0.9982380198765312, 0.8995147229230275],
[0.998020889041353, 0.8971381683711599],
[0.9977905543165964, 0.8948493824527858],
[0.9975476773075732, 0.8926495731831078],
[0.997293047809275, 0.8905396026735302],
[0.9970275715061022, 0.8885199936961394],
[0.9967522557819561, 0.886590937644079],
[0.9964681940519674, 0.8847523040841261],
[0.9961765490509034, 0.8830036520098776],
[0.9958785355202159, 0.8813442428229699],
[0.9955754027265399, 0.8797730549977745],
[0.995268417220794, 0.878288800323429],
[0.9949588462110325, 0.8768899415665314],
[0.9946479418764367, 0.8755747113585308],
[0.9943369268971146, 0.8743411320834718],
[0.9940269814175402, 0.8731870365236551],
[0.9937192316031929, 0.87211008901205],
[0.9934147398927153, 0.871107806839864],
[0.9931144969937427, 0.870177581674368],
[0.9928194156211504, 0.8693167007546527],
[0.9925303259330255, 0.8685223676502678],
[0.9922479725829699, 0.8677917223884959],
[0.9919730132777788, 0.8671218607792585],
[0.9917060187071102, 0.8665098527913737],
[0.9914474736962087, 0.8659527598592116],
[0.99119777942355, 0.8654476510239902],
[0.9909572565417608, 0.8649916178383918],
[0.9907261490415585, 0.8645817879863853],
[0.9905046287039136, 0.864215337591701],
[0.9902927999943216, 0.8638895022080644],
[0.9900907052641822, 0.8636015865018694],
[0.9898983301370634, 0.8633489726533542],
[0.9897156089714192, 0.8631291275155126],
[0.9895424303055419, 0.8629396085809611],
[0.9893786422046935, 0.8627780688158673],
[0.9892240574440706, 0.8626422604269535],
[0.9890784584742288, 0.8625300376326458],
[0.9889416021276027, 0.8624393585128404],
[0.9888132240356547, 0.8623682860136488],
[0.988693042735909, 0.8623149881840648],
[0.9885807634566216, 0.8622777377209356],
[0.9884760815741404, 0.8622549108971003],
[0.9883786857441481, 0.8622449859452446],
[0.9882882607130381, 0.8622465409670598],
[0.9882044898197255, 0.8622582514338453],
[0.988127057201344, 0.8622788873408672],
[0.988055649718626, 0.8623073100737135],
[0.9879899586184036, 0.8623424690406561],
[0.9879296809517036, 0.8623833981207344],
[0.9878745207664422, 0.8624292119729884],
[0.987823909219373, 0.8624780721239566],
[0.9877776002822488, 0.8625293387039025],
[0.9877353524392587, 0.862582432558867],
[0.9876969297050266, 0.8626368309979183],
[0.9876621023804052, 0.8626920636382249],
[0.9876306475643788, 0.8627477082561527],
[0.9876023494869289, 0.8628033867476519],
[0.987576999718532, 0.8628587612932762],
[0.987554397301005, 0.862913530804051],
[0.9875343488328757, 0.8629674276989803],
[0.9875343488328757, 0.8629674276989803]]
            
            EmsplexWithLambda4 = np.array(EmsplexWithLambda4)
            EmsplexHalfDetailed = np.array(EmsplexHalfDetailed)
            EmsplexAvailableProbabilities = np.array(EmsplexAvailableProbabilities)
            Emsplex10timesMoreDetailed = np.array(Emsplex10timesMoreDetailed)
            for b in range(2):
                if CALL_LAMBDA == 4/60:
                    plt.plot(time, EmsplexWithLambda4[:, b], label=f"Qplex") #with Δt = 1
                else:
                    plt.plot(time, EmsplexHalfDetailed[:, b], label=f"Qplex with Δt = 2")
                    plt.plot(time, Emsplex10timesMoreDetailed[:, b], label=f"Qplex with Δt = 0.1")
                    plt.plot(time, EmsplexAvailableProbabilities[:, b], label=f"Qplex with Δt = 1")
            plt.xlabel("Time")
            plt.ylabel("Probability")
            plt.title(f"P(vehicle available at base) with 95% CI and {CALL_LAMBDA*60} calls per hour")
            plt.legend()
            plt.grid(True)
            plt.show()
            
            np.set_printoptions(threshold=1000)  #restore default behavior afterward

        CI_error_m_mean_response_times = scipy.stats.t.ppf(
            0.975, NUM_RUNS - 1
        ) * (np.std(mean_response_times, ddof=1) / np.sqrt(NUM_RUNS))
        print(
            "The 95% CI of the mean mean response time is:"
            f"({m_mean_response_times-CI_error_m_mean_response_times},"
            f"{m_mean_response_times+CI_error_m_mean_response_times})."
        )

    print(
        "The mean 95% empirical quantile of the response time over "
        f"all runs is: {m_emp_quantile_response_times}."
    )
    if NUM_RUNS > 1:
        CI_error_m_emp_quantile_response_times = scipy.stats.t.ppf(
            0.975, NUM_RUNS - 1
        ) * (np.std(emp_quantile_response_times, ddof=1) / np.sqrt(NUM_RUNS))
        print(
            "The 95% CI of the mean 95% empirical quantile of the response time is:"
            f"({m_emp_quantile_response_times-CI_error_m_emp_quantile_response_times},"
            f"{m_emp_quantile_response_times+CI_error_m_emp_quantile_response_times})."
        )

    print(f"The mean busy fraction over all runs is: {m_busy_fractions}.")
    if NUM_RUNS > 1:
        CI_error_m_busy_fractions = scipy.stats.t.ppf(0.975, NUM_RUNS - 1) * (
            np.std(busy_fractions, ddof=1) / np.sqrt(NUM_RUNS)
        )
        print(
            "The 95% CI of the mean busy fraction is:"
            f"({m_busy_fractions-CI_error_m_busy_fractions},"
            f"{m_busy_fractions+CI_error_m_busy_fractions})."
        )

    if SIMULATION_PARAMETERS["SAVE_OUTPUT"]:
        pd.DataFrame(mean_response_times).to_csv(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['MEAN_RESPONSE_TIMES_FILE_NAME']}.csv"
        )
        pd.DataFrame(emp_quantile_response_times).to_csv(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['EMP_QUANTILE_RESPONSE_TIMES_FILE_NAME']}.csv"
        )
        pd.DataFrame(busy_fractions).to_csv(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['BUSY_FRACTIONS_FILE_NAME']}.csv"
        )
        pd.DataFrame(
            {
                "run_nr": np.arange(NUM_RUNS),
                "Running_time (sec)": running_times,
            }
        ).to_csv(
            f"{SIMULATION_PARAMETERS['SIMULATION_OUTPUT_DIRECTORY']}"
            f"{SIMULATION_PARAMETERS['RUNNING_TIME_FILE_NAME']}.csv"
        )

    for key in copy_simulation_parameters.keys():
        if copy_simulation_parameters[key] != SIMULATION_PARAMETERS[key]:
            raise Exception(
                "The SIMULATION_PARAMETERS were altered during "
                "the simulation. This should not happen. Error."
            )

    print(
        "\nRunning the complete main.py script takes: "
        f"{datetime.datetime.now()-start_time_script}."
    )

    print("\007")
