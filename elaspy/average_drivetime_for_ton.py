import os
import sys
import pandas as pd

ROOT_DIRECTORY: str = os.path.dirname(os.path.dirname(__file__))
DATA_DIRECTORY: str = os.path.join(ROOT_DIRECTORY, "data/") #this is utrecht

TRAVEL_TIMES_FILE: str = "siren_driving_matrix_2022.csv"
NODES_FILE: str = "nodes_Utrecht_2021.csv"

#I did use the following files, but put the info in there hardcoded in this file
#HOSPITAL_FILE: str = "Hospital_Postal_Codes_Utrecht_2021.csv"
#BASE_LOCATIONS_FILE: str = "RAVU_base_locations_Utrecht_2021.csv"
#AMBULANCE_BASE_LOCATIONS_FILE: str = ("Base_Locations.csv")

TRAVEL_TIMES_DF = pd.read_csv(  f"{DATA_DIRECTORY}{TRAVEL_TIMES_FILE}",        index_col=0,    )
TRAVEL_TIMES_DF.index = TRAVEL_TIMES_DF.index.astype(int)
TRAVEL_TIMES_DF.columns = TRAVEL_TIMES_DF.columns.astype(int)

def lookUpDriveTime(fromPC, toPC, switch=True): #switch interprets the matrix tranposed   
   drive_time = 0
   if switch:
       drive_time = TRAVEL_TIMES_DF.loc[toPC,fromPC] 
   else:
       drive_time = TRAVEL_TIMES_DF.loc[fromPC, toPC] 
   #sys.exit(f"Drive time from {fromPC} to {toPC}: {drive_time:.2f} minutes")
   return drive_time 

def DriveTimeToNearestHospital(demandNodePC):        
    hospitalPCs = [3584,3435,3543,3582,3813]
    result = 99999999999
    for hosPC in hospitalPCs:
        drivTime = lookUpDriveTime(demandNodePC, hosPC) 
        if drivTime < result:
            result = drivTime
    return result

def getMeanDriveTimeToNearestHospital():
    result = 0
    df = pd.read_csv( f"{DATA_DIRECTORY}{NODES_FILE}")
    inhabitants = df['inhabitants'].astype(float).to_numpy()
    postCodes = df['postal code'].astype(int).to_numpy()  
    for i in range(len(inhabitants)):
        result += inhabitants[i]*DriveTimeToNearestHospital(postCodes[i])
    return result

def DriveTimeFromNearestBase(demandNodePC):        
    BasePCs = [3812,3821,3823,3941,3608,3435,4145,3582,3561,3911,3645,3447,3707,3417,3648,3743,3769,3931,3958,3991,4128]
    result = 99999999999
    for basePC in BasePCs:
        drivTime = lookUpDriveTime(basePC,demandNodePC) 
        if drivTime < result:
            result = drivTime
    return result

def getMeanDriveTimeFromNearestBase():
    df = pd.read_csv( f"{DATA_DIRECTORY}{NODES_FILE}")
    inhabitants = df['inhabitants'].astype(float).to_numpy()#tested this works
    postCodes = df['postal code'].astype(int).to_numpy()  
    result = 0
    for i in range(len(inhabitants)):
        result += inhabitants[i] * DriveTimeFromNearestBase(postCodes[i])
    return result


if __name__ == "__main__":
    #calculate_drivetime_to_add_to_service_times:
    print(f'MeanDriveTimeFromNearestBase={getMeanDriveTimeFromNearestBase()} and MeanDriveTimeToNearestHospital={getMeanDriveTimeToNearestHospital()}')
    print(f'result = {getMeanDriveTimeFromNearestBase() + 0.63 * getMeanDriveTimeToNearestHospital()}')
    