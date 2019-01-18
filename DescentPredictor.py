import pystan
from itertools import tee, chain
import pandas as pd
import sqlite3 as sql
from TimeSeries0 import lag, factor_lag, lookup_previous
import datetime as dt
import numpy as np
import pickle
import scipy.stats as stats



class FlightData():
    def __init__(self, datasource, datasource_type="csv"):
        self.datasource = datasource
        self.datasource_type = datasource_type

        #get the data from the data source
        if datasource_type=="csv":
            self.data=pd.read_csv(datasource)
        elif datasource_type=="sql":
            self.data=pd.read_sql(datasource)
        elif datasource_type=="dict":
            self.data=pd.DataFrame(datasource)
        else:
            raise ValueError("Datasource of unkown type.")

    def get_update(self):
        #this reloads the data based on the settings on creation.
        #mostly I wanted to simplify auto_update
        self.__init__(self.datasource, self.datasource_type)
        #I should probably add some error checking



#this is not used anywhere
class PredictedPath():
    '''
    This class builds a predicted path given some parameters and a model.
    Hopefully we can move the generate predictions function below to this
    '''
    def __init__(self, ground_altitude, flight_data, parameters):
        #ground_altitude is the altitude above sea level in the area
        #altitude_hist is the altitude history
        #parameter_model is a function that takes some given parameters and altitude_hist
        #parameters is the parameters for parameter_model
        self.ground_altitude = ground_altitude

        self._predict()

    def __str__(self):
        return self.path

    def predict(self):
        pass


def generate_prediction(data,
                        parameters,
                        sample_size=1000,
                        ground_altitude = 5000*12/39,
                        max_altitude =   4e4, #just in case the prediction starts us going up
                        parameter_resample=True,
                        vel_true=False):
    '''
    This generates a datatable of predicted flight paths
    '''
    #build datastructure and get a sample set of parameters.
    list_of_predictions= {}
    #sample the given parameters
    tpam=parameters.sample(n=sample_size, replace=True)

    #For each parameter set (row)
    for i,row in enumerate(tpam.itertuples()):
        #Get flight history data
        test_flight_hist = data[-2:]

        #build a new (properly adjusted) iterator
        prevs,items = tee(test_flight_hist,2)
        prevs = chain([None],prevs)

        #for each entry in the new iterator
        for prev,current in zip(prevs,items):
            #Do sanity checks
            if prev and prev > max_altitude:
                #print("Above max altitude Altitude")
                break
            elif current < ground_altitude:
                #print("Below Ground Altitude")
                break
            elif current and prev:
                #calculate velocity.
                # NOTE: in the Descent Modeler velocity is calculated as
                velocity = current-prev
                #velocity = prev-current

                #Calculate mu, and then select a prediction
                if vel_true:
                    mu = row.lag_1*current + row.lag_2*prev + row.alphas + row.velocity*prev
                else:
                    mu = row.lag_1*current + row.lag_2*prev + row.alphas #+ row.velocity*prev
                    #print(mu)
                    pred = stats.norm.rvs(loc=mu, scale=row.stdev)

                    #save the prediction
                    test_flight_hist.append(pred)
                    if i%100==0:
                        print("PATH #{} added".format(i))

                        list_of_predictions[row] = pd.Series(test_flight_hist)

    return pd.DataFrame.from_dict(list_of_predictions)



"""
Predictive values.
Use the given data to simulate various paths to the ground.
I have chosen to program the actual prediction in python. The reason for this
is that it will be easier to dynamically update the results by doing so.
"""
quantile_list = [0.025, 0.25,0.5,0.75,0.975]

#do we want to include the velocity as part of the model?
vel_true = False


#Get parameters
parameters = pd.read_pickle("./DataModel.pkl")

print("Parameter Columns")
print(parameters.columns)
print("Parameter Quantiles")
print(parameters.quantile(quantile_list))
print(parameters.mean())

if input("Continue (enter) or quit (q)")=="q":
    exit()



#Create the last few altitudes
flight_data = [33131, 28030, 23631]


predictions=generate_prediction(flight_data, parameters)
predictions.to_csv("predictions.csv")

pred_length=predictions.count().quantile(quantile_list)

print("Predicted length in minute quantiles")
print(pred_length/2)

"""
Display Module
I hope to be able to generate images representing the probablility of landing
in different areas.
"""
#The plan is to interpret this data in bokeh
