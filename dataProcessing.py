import os

import pandas as pd
import numpy as np
from tensorflow.python import train
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
import joblib

from parameters import COMPANY, DATA_SOURCE, END_DATE, START_DATE

# ---------------------------------------------------------------------------------------------------------------------------------------------------

def processData(isStoredDataLocally=True, company="TSLA", startDate="2015-01-01", endDate="2020-01-01", dataSource = "yahoo", 
                    trainRatio=0.8, randomSplit=False, randomSeed=None,
                    isScaledData=True, featureRange=(0, 1), isStoredScaler=True):
    
    #   Check if the stock data's folder is exist
    if (not os.path.isdir("stockdatas")):
        os.mkdir("stockdatas")    

    # LOAD DATA
    trainDataFilePath = "stockdatas/{}_{}.csv".format(company, "training")
    testDataFilePath = "stockdatas/{}_{}.csv".format(company, "testing")

    if (dataSource == "yahoo"):
        # Check if the data has been saved
        if (os.path.exists(trainDataFilePath) and (os.path.exists(testDataFilePath))):
            trainData = pd.read_csv(trainDataFilePath, index_col=0, parse_dates=True)
            testData = pd.read_csv(testDataFilePath, index_col=0, parse_dates=True)
            print("Loaded datas from the saved file")
        # If the data has not been saved
        else:
            # Fetch data from Yahoo Finance
            data = yf.download(company, start=startDate, end=endDate, progress=False)
            usedData = data.copy()                                             # Make a copy to avoid modifying the original data
            # Handle NaN values in the data
            usedData.fillna(method='ffill', inplace=True)                      # Forward fill missing values

            # SPLIT DATA
            (trainData, testData) = splitData(usedData, trainRatio, randomSplit, randomSeed)
            print("Fetched data.")

            # SAVE DATA
            if (isStoredDataLocally):
                trainData.to_csv(trainDataFilePath)
                testData.to_csv(testDataFilePath)

        # SCALE DATA 
        if (isScaledData):
            trainData = scaleData(company, trainData, "training", isStoredScaler, featureRange)
            testData = scaleData(company, testData, "testing", isStoredScaler, featureRange)

        return (trainData, testData)

def splitData(data, trainRatio, randomSplit, randomSeed):
    if (not 0 < trainRatio < 1):
        raise ValueError("train ratio should be between 0 and 1.")    
    
    # Split randomly
    if (randomSplit):
        if (randomSeed is not None):
            np.random.seed(randomSeed)
        mask = np.random.rand(len(data)) < trainRatio
        trainData = data[mask]
        testData = data[~mask]
    # Split by date: The training data comes from the earlier dates and the testing data comes from the later dates
    else:
        splitDate = int(len(data) * trainRatio)
        trainData = data[:splitDate]
        testData = data[splitDate:]
    
    return trainData, testData

def scaleData(company, data, dataType, isStoredScaler, featureRange=(0, 1)):
    # Check if the stock data's folder is exist
    if (not os.path.isdir("stockdatas")):
        os.mkdir("stockdatas")    

    scalerFilePath = "stockdatas/scaler_{}_{}.pkl".format(company, dataType)
    dataFilePath = "stockdatas/{}_{}.csv".format(company, dataType)
    
    # No data found
    if (not os.path.exists(dataFilePath)):
        print("The {} data is missing.".format(dataType))
    # Scale the data with the new scaler
    elif (not os.path.exists(scalerFilePath)):
        # Scale data
        scaler = MinMaxScaler(feature_range=featureRange)
        scaledData = scaler.fit_transform(data)
        print("Scaled {} data.".format(dataType))
        
        # Save scaler
        if (isStoredScaler):
            joblib.dump(scaler, scalerFilePath)
            print("Saved {} scaler.".format(dataType))
    # Load scaler and scaled data
    else:
        data = pd.read_csv(dataFilePath, index_col=0, parse_dates=True)
        scaler = joblib.load(scalerFilePath)
        print("Loaded {} scaler.".format(dataType))
        
        scaledData = scaler.transform(data.copy())  
        print("Scaled {} data.".format(dataType))

    return scaledData, scaler

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST DATA PROCESSING

# (trainData, testData) = processData(True, COMPANY, START_DATE, END_DATE, DATA_SOURCE, 0.8, False, None, True, (0, 1), True)

# print("===============================================================")
# print("Train Data:")
# print(trainData)

# print("===============================================================")
# print("Test Data")
# print(testData)