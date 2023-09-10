import collections
import os
from unittest import result
import joblib

import pandas as pd
import numpy as np
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from parameters import *

# ---------------------------------------------------------------------------------------------------------------------------------------------------

def processData(isStoredDataLocally=True, company="TSLA", startDate="2015-01-01", endDate="2020-01-01", dataSource = "yahoo", 
                    predictionDays=50, lookupSteps=1, featureColumns=["Open", "High", "Low", "Close"],
                    trainRatio=0.8, randomSplit=False, randomSeed=None,
                    isScaledData=True, featureRange=(0, 1), isStoredScaler=True):
    
    #   Check if the stock data's folder is exist
    if (not os.path.isdir("stockdatas")):
        os.mkdir("stockdatas")    

    # LOAD DATA
    dataFileName = "{}-company_{}-type_{}-startD_{}-endD".format(company, "data", startDate, endDate)
    dataFilePath = "stockdatas/{}.csv".format(dataFileName)

    processedData = {}

    if (dataSource == "yahoo"):
        # Check if the data has been saved
        if (os.path.exists(dataFilePath)):
            data = pd.read_csv(dataFilePath, index_col=0, parse_dates=True)
            print("Loaded data from the saved file")
        # If the data has not been saved
        else:
            # Fetch data from Yahoo Finance
            data = yf.download(company, start=startDate, end=endDate, progress=False)
            
            # Handle NaN values in the data by forwarding fill missing values
            data.fillna(method='ffill', inplace=True)   
            
            # Save data
            if (isStoredDataLocally):
                data.to_csv(dataFilePath)                   

        processedData["Data"] = data
        
        # Add date as a column
        if ("Date" not in data.columns):
            data["Date"] = data.index

        # SCALE DATA 
        if (isScaledData):
            (data, featureColumnScalers) = scaleData(data=data, dataFileName=dataFileName, featureColumns=featureColumns, featureRange=featureRange, isStoredScaler=isStoredScaler)
            
            processedData["ColumnScalers"] = featureColumnScalers
            
        # Add the target column (label) by shifting by <lookupStep>
        data["Future"] = data["Close"]
        
        # The last <lookupSteps> column(s) contains NaN in the "Future" column, get them before droping NaNs
        lastSequence = np.array(data[featureColumns].tail(lookupSteps))
    
        # Handle NaN values in the data by forwarding fill missing values
        data.fillna(method='ffill', inplace=True)   
        
        # Create a list of sequences of features and their corresponding targets
        sequenceData = []
        sequences = collections.deque(maxlen=predictionDays)
        
        for (sequence, target) in zip(data[featureColumns + ["Date"]].values, data["Future"].values):
            sequences.append(sequence)      

            if (len(sequences) == predictionDays):
                sequenceData.append([np.array(sequences), target])

        # Get the last sequence by appending the last <predictionDays> sequence with <lookupSteps> sequence
        # E.g. If lookupSteps=a and predictionDays=b, lastSequence should be of (a+b) length
        # This lastSequence will be used to predict future stock prices that are not available in the dataset
        lastSequence = list([sequence[:len(featureColumns)] for sequence in sequences]) + list(lastSequence)     

        lastSequence = np.array(lastSequence).astype(np.float32)
        
        processedData["LastSequence"] = lastSequence
        
        # Construct the x (sequences) and y (targets)
        (x, y) = ([], [])
        for (sequence, target) in sequenceData:
            x.append(sequence)
            y.append(target)
            
        # Convert to Numpy arrays
        x = np.array(x)
        y = np.array(y)
        
        # SPLIT DATA
        (processedData["XTrain"], processedData["XTest"], processedData["YTrain"], processedData["YTest"]) = splitData(x=x, y=y, 
                                                                                                                       trainRatio=trainRatio, randomSplit=randomSplit, randomSeed=randomSeed)   

        # Get the xTest dates
        dates = processedData["XTest"][:, -1, -1]
        # Retrieve test features dates from the original data
        processedData["TestData"] = processedData["Data"].loc[dates]
        # Remove dupliacate dates
        processedData["TestData"] = processedData["TestData"][~processedData["TestData"].index.duplicated(keep="first")]
        # Remove dates from xTrain and xTest and convert to float32
        processedData["XTrain"] = processedData["XTrain"][:, :, :len(featureColumns)].astype(np.float32)
        processedData["XTest"] = processedData["XTest"][:, :, :len(featureColumns)].astype(np.float32)
        
        return processedData

def splitData(x, y, trainRatio=0.8, randomSplit=False, randomSeed=None):
    if (not 0 < trainRatio < 1):
        raise ValueError("train ratio should be between 0 and 1.")   
    
    if (randomSplit):
        (xTrain, xTest, yTrain, yTest) = train_test_split(x, y, test_size=(1 - trainRatio), shuffle=False)
    else:
        trainSamples = int(trainRatio * len(x))
        
        (xTrain, xTest, yTrain, yTest) = (x[:trainSamples], x[trainSamples:], y[:trainSamples], y[trainSamples:])

    return (xTrain, xTest, yTrain, yTest)

def scaleData(data, dataFileName, featureColumns=["Open", "High", "Low", "Close"], featureRange=(0, 1), isStoredScaler=True):
    # Check if the stock data's folder is exist
    if (not os.path.isdir("stockdatas")):
        os.mkdir("stockdatas")
        
    scaledData = data.copy()
    featureColumnScalers = {}
    
    for col in featureColumns:
        scalerFilePath = "stockdatas/{}_scaler_{}.pkl".format(col, dataFileName)        

        # Scale the data with the new scaler
        if (not os.path.exists(scalerFilePath)):
            # Scale data
            print(data[col])
            print("=====================")

            scaler = MinMaxScaler(feature_range=featureRange)
            
            data[col] = scaler.fit_transform(np.expand_dims(data[col], axis=1))

            print("Scaled data.")
       
            # Save scaler
            if (isStoredScaler):
                joblib.dump(scaler, scalerFilePath)
                print("Saved scaler.")
        # Load scaler and scaled data
        else:
            scaler = joblib.load(scalerFilePath)
            print("Loaded scaler.")
        
            scaledData[col] = scaler.transform(np.expand_dims(data[col], axis=1))  
            print("Scaled data.")
            
        featureColumnScalers[col] = scaler

    return scaledData, featureColumnScalers

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST DATA PROCESSING

# processedData = processData(isStoredDataLocally=True, company=COMPANY, startDate=START_DATE, endDate=END_DATE, dataSource=DATA_SOURCE, 
#                             predictionDays=PREDICTION_DAYS, lookupSteps=LOOKUP_STEPS, featureColumns=FEATURE_COLUMNS,
#                             trainRatio=0.8, randomSplit=False, randomSeed=None, isScaledData=True, featureRange=(0, 1), isStoredScaler=True)

# print("===================== DATA =======================")
# print(processedData["Data"])
# print("================= LAST SEQUENCE ==================")
# print(processedData["LastSequence"])
# print("==================== X-TRAIN =====================")
# print(processedData["XTrain"])
# print("==================== Y-TRAIN =====================")
# print(processedData["YTrain"])
# print("==================== X-TEST ======================")
# print(processedData["XTest"])
# print("==================== Y-TEST ======================")
# print(processedData["YTest"])

