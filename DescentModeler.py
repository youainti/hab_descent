"""
This program will take data from the database, the stan model file,
and the last known flight data to build a descent prediction.
"""
import pystan
import pandas as pd
import sqlite3 as sql
from TimeSeries0 import lag, factor_lag, lookup_previous
import datetime as dt
import numpy as np
import pickle


vel_true = False

"""
Import and modify data
This is where the data will be imported and adjusted to the format that is
needed.
"""
#import data from database
db_con = sql.connect("./FlightPredictor.db")


date_list = ["time", "lasttime"]
#ascent profile for lookups
ascent_text = r'select * from aprs_fi where profile="Ascent";'
ascent = pd.read_sql_query(ascent_text, db_con,
                           parse_dates=date_list)

#descent profile for model
descent_text = 'select * from aprs_fi where profile="Descent";'
descent = pd.read_sql_query( descent_text,db_con,
                            parse_dates=date_list)


#Set the index for both
#ascent = ascent.set_index('time')
#descent = descent.set_index('time')

descent_profile = pd.merge_asof(descent.sort_values(by='altitude'),
                     ascent.sort_values(by='altitude'),
                     on='altitude',
                     by='flightnum',
                     direction='nearest',
                     suffixes=('_d', '_a'))


descent_profile = descent_profile.set_index('time_d')

#fill in all gaps in ascent and descent profiles.
#lag appropriate variables
lagged = factor_lag(descent_profile,
           factors=["callsign_d","flightnum"],
           columns=["lat_d","lng_d","altitude","lat_a","lng_a"],
           times=[-1,1,2],
           resample_period='30s',
           drop_na=True)


#Add velocity measures
lagged["altitude_velocity"] = lagged.altitude_lag1 - lagged.altitude_lag2

#Subset the data
adjusted_data = lagged[["altitude",
                        "altitude_lag1","altitude_lag2","lat_d",
                        "lat_d_lag1","lat_d_lag2","lng_d",
                        "lng_d_lag1","lng_d_lag2","lat_a",
                        "lat_a_lag1","lng_a_lag1","lat_a_lag2","lng_a",
                        #"lng_d_lead1",
                        #"altitude_lead1",
                        #"lat_d_lead1",
                        #"lat_a_lead1",
                        #"lng_a_lead1",
                        "lng_a_lag1","lng_a_lag2"]]
print(adjusted_data.columns)
altitude_data= lagged[["altitude",
                       "altitude_lag1",
                       "altitude_lag2",
                       "altitude_velocity"]]
print(altitude_data.columns)

"""
Statistical Model
This is where the stan model will be built and run.
"""
#check if a compiled model exists
    #if not, compile it
default_model="./MLR-flightpredictor.stan"
default_model_bin = default_model+".pkl"
try:
    #read stan model
    model_code = pickle.load(open(default_model_bin, "rb") )
except:
    #Given no stan model binary exists, create a new one
    print("Precompiled Model not found, generating new model.")
    model_code = pystan.StanModel(file=default_model)

    with open(default_model_bin, "wb") as f:
        pickle.dump(model_code, f)

    print("Model generated and saved")

#Run the model using the data given...
if vel_true:
    model_data = {
        "N": len(altitude_data.altitude),
        "k": len(altitude_data.columns)-1,
        "alt_X": altitude_data[["altitude_lag1", 'altitude_lag2',"altitude_velocity"]],
        "alt_y": altitude_data.altitude}
else:
    model_data = {
        "N": len(altitude_data.altitude),
        "k": len(altitude_data.columns)-2,
        "alt_X": altitude_data[["altitude_lag1", 'altitude_lag2']],
        "alt_y": altitude_data.altitude}
#print(model_data)
quantile_list = [0.025, 0.25,0.5,0.75,0.975]

fit = model_code.sampling(data=model_data,
                          iter=10000,
                          warmup=1000,
                          chains=4)

fitted = fit.extract()
if vel_true:
    param_values = pd.DataFrame(fitted["beta"], columns=("lag_1", 'lag_2', "velocity"))
else:
    param_values = pd.DataFrame(fitted["beta"], columns=("lag_1", 'lag_2'))
param_values["alphas"] = pd.DataFrame(fitted["alpha"])
param_values["stdev"] = pd.DataFrame(fitted["stdev"])

export_params_file="./DataModel.pkl"
param_values.to_pickle(export_params_file)

print("Parameter Values exported to: {}".format(export_params_file))
