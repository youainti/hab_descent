'''
This is a set of functions that can be used for working with TimeSeries data
'''
import pandas as pd
import datetime as dt

def lag(data, column, times=1, drop_na=True):
    """
    Implements lagging a variable by 'number' observations
    Parameters
        Column = string or list of strings, where each string is a column.
        times = number of periods by which the column should be lagged.
        drop_na = True (Default) means that NA/NaN values should be dropped.
            False would mean that the DataFrame does not have these values
            dropped.
    """
    if type(times) != type(1):
        raise ValueError("times is not an int.")

    if type(column) == type(""):
        #make sure the value is valid
        if column not in data:
            raise ValueError("Column not found")
        #lag the variable
        new_title = "{}_lag{}".format(column,times)
        data[new_title] = data[column].shift(times)

    elif type(column) == type([]): #if you get a list of columns
        for item in column:
            #Make sure each option is in the list of columns
            if item not in data:
                raise ValueError("Column not found.")
            #Lag the variable
            if times > 0:
                new_title = "{}_lag{}".format(item,times)
            else:
                new_title = "{}_lead{}".format(item,-times)
            data[new_title] = data[item].copy().shift(times)
    else:
        raise ValueError("Column parameter must be either a string or list of strings, where each string is a column name.")


    if drop_na:
        data = data.dropna()

    return data



def factor_lag(data, factors, columns, times=[1], drop_na=False, resample_period=None):

    #base case
    if len(factors) == 0:
        #interpolate factors
        if type(resample_period) == type(''):
            data = data.resample(resample_period).last()
            data = data.interpolate()
            data = data.fillna(method='ffill')

        #lag multiple times
        for lagged in times:
            data = lag(data, column=columns, times=lagged, drop_na=False)
        return data

    else: #if there are factors left to parse, then keep going
        #check factors
        factor = factors[0]
        if factor not in data.columns:
            raise ValueError("Factor not Found")
        else:
            #apply first factor subset
            level_data = []
            for level in data[factor].unique():
                #subset the data
                subset = data[data[factor]==level]
                #recurse
                level_datum = factor_lag(subset,
                                         factors=factors[1:],
                                         columns=columns,
                                         times=times,
                                         drop_na=False,
                                         resample_period=resample_period)

                #take the given data, and add it together
                level_data.append(level_datum)


            if drop_na:
                return pd.concat(level_data, axis=0, join='outer').dropna()
            else:
                return pd.concat(level_data, axis=0, join='outer')

def lookup_previous(data, lookup_column, lookup_value,
                  return_columns=[], filter=(None,None)):
    '''
    This will lookup a given value in a given column, and return
    number_results obervations from return_columns
    '''
    #find the values below
    if len(filter) > 2 or type(filter) != type(()):
        raise ValueError("Error with filter. Expected Tuple of size 2, got {} of size {}".format(type(filter),len(filter)))

    if filter[0] and filter[1]:
        subset = data[data[filter[0]] == filter[1]]
        subset = subset[subset[lookup_column] <= lookup_value]
        maxed = subset[lookup_column].max()
        subset = subset[subset[lookup_column] == maxed]
    elif not filter[0] and not filter[1]:
        subset = data[data[lookup_column] <= lookup_value]
        maxed = subset[lookup_column].max()
        subset = subset[subset[lookup_column] == maxed]
    else:
        raise ValueError("filter not recognized.")


    if not return_columns:
        return_columns=lookup_column

    return subset[return_columns].head(1)



def main():
    #testing
    d = {"a":[0,1,2,3,4,-1],
         "b":[5,6,7,8,9,10] ,
         "c":["q","q","q", "r","r","r"],
         "d":[1,1,1,2,2,2]}
    s = pd.DataFrame(d)
    lag(s, column=["a", "b"])

    if not True:
        print(s)
        print(s)


    s2 = factor_lag(s, ["c","d"], columns=["b", "a"])
    s3 = factor_lag(s2, ["c","d"], columns=["b", "a"], times=[1], drop_na=True)
    if not True:
        print(s2)
        print(s3)

    if not True:
        test=lookup_previous(s, lookup_column='b', lookup_value=6, return_columns=["b","c"])
        print(test)

    if True:
        for i in s.b:
            test = lookup_previous(s,'b',i)
            print(test)
            s["test"] = test
            print(s)

if __name__ == "__main__":
    main()
