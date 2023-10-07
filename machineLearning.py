import itertools
import os
import joblib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import layers, models, metrics
from pmdarima import auto_arima
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from parameters import *
from dataProcessing import *
from dataVisualizing import *

# ---------------------------------------------------------------------------------------------------------------------------------------------------

def constructDLModel(featureColumns=["Open", "High", "Low", "Close"], numOfPastDays=50, numOfFutureDays=1, layersNumber=2, layerSize=256, layerName=layers.LSTM, 
                     dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    # Check if the layersNumber >= 1
    if (layersNumber < 1):
        raise ValueError("Number of layers should be equal to or more than 1")

    # Initialize a Sequential model
    model = models.Sequential()    

    for i in range(layersNumber):
        # First Layer
        if (i == 0):
            if (bidirectional):
                model.add(layers.Bidirectional(layerName(units=layerSize, return_sequences=True), input_shape=(numOfPastDays, len(featureColumns))))
            else:
                model.add(layerName(units=layerSize, return_sequences=True, input_shape=(numOfPastDays, len(featureColumns))))
        # Last layer
        elif (i == layersNumber - 1):
            if (bidirectional):
                model.add(layers.Bidirectional(layerName(units=layerSize, return_sequences=False)))
            else:
                model.add(layerName(units=layerSize, return_sequences=False))
        # Other layers
        else:
            if (bidirectional):
                model.add(layers.Bidirectional(layerName(units=layerSize, return_sequences=True)))
            else:
                model.add(layerName(units=layerSize, return_sequences=True))
                
        # Add Dropout after each layer
        model.add(layers.Dropout(dropout))
    
    # Add Dense output layer (with <NUMBER_OF_FUTURE_DAYS> neuron (unit))
    model.add(layers.Dense(numOfFutureDays))
    
    # Compile the model (with loss function, evaluation metric (mean_absolute_error), and optimizer)
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=[metrics.MeanAbsoluteError()])
    
    return model

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST MACHINE LEARNING MODEL - MULTIVARIATE PROBLEM
def trainAndTestMultivariateDLModel(processedData, featureColumns=["Open", "High", "Low", "Close"], numOfPastDays=50, numOfFutureDays=1, layersNumber=2, layerSize=256, layerName=layers.LSTM, 
                                 dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False,
                                 epochs=50, batchSize=32):
    # Training
    #   Check if the model's folder is exist
    if (not os.path.isdir("models")):
        os.mkdir("models")  
        
    modelFilePath = "models/{}-fCols_{}-pDays_{}-fdays_{}-layers_{}-lSize_{}-lName_{}-do_{}-loss_{}-optimizer_{}biD_{}-epochs_{}-bSize.keras".format(len(featureColumns), numOfPastDays, numOfFutureDays, layersNumber, 
                                                                                                                     layerSize, str(layerName).split('.')[-1].split("'")[0], 
                                                                                                                     dropout, loss, optimizer, bidirectional, epochs, batchSize)

    if (not os.path.exists(modelFilePath)):        
        # Create and train the Deep Learning model
        model = constructDLModel(featureColumns=featureColumns, numOfPastDays=numOfPastDays, numOfFutureDays=numOfFutureDays, layersNumber=layersNumber, layerSize=layerSize, layerName=layerName, 
                                 dropout=dropout, loss=loss, optimizer=optimizer, bidirectional=bidirectional)

        model.fit(x=processedData["XTrain"], y=processedData["YTrain"], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(processedData["XTest"], processedData["YTest"]))
        
        # Save the model
        model.save(modelFilePath)
        print("Trained model saved to {}".format(modelFilePath))
    else:
        # Load the model
        model = models.load_model(modelFilePath)
        print("Trained model loaded from {}".format(modelFilePath))
        
    # Testing
    #   Get actual and predicted data
    yActualData = processedData["YTest"]
    yPredictedData = model.predict(processedData["XTest"])

    #   Descale
    if (IS_SCALED_DATA):
        yActualData = np.squeeze(processedData["ColumnScalers"]["Close"].inverse_transform(yActualData))
        yPredictedData = np.squeeze(processedData["ColumnScalers"]["Close"].inverse_transform(yPredictedData))
        
    # print("================ Y-ACTUAL DATA ===================")
    # print(yActualData)
    # print("=============== Y-PREDICTED DATA =================")   
    # print(yPredictedData)
  
    #   Plot
    # plotSingleFeature("{} Close Price Multivariate Prediction", yActualData, yPredictedData)
    
    return (yActualData, yPredictedData)

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST MACHINE LEARNING MODEL - MULTISTEP + MULTIVARIATE PROBLEM

def trainAndTestMultistepDLModel(processedData, featureColumns=["Open", "High", "Low", "Close"], numOfPastDays=50, numOfFutureDays=1, layersNumber=2, layerSize=256, layerName=layers.LSTM, 
                                 dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False,
                                 epochs=50, batchSize=32):
    # Training
    #   Check if the model's folder is exist
    if (not os.path.isdir("models")):
        os.mkdir("models")  
        
    modelFilePath = "models/{}-fCols_{}-pDays_{}-fdays_{}-layers_{}-lSize_{}-lName_{}-do_{}-loss_{}-optimizer_{}biD_{}-epochs_{}-bSize.keras".format(len(featureColumns), numOfPastDays, numOfFutureDays, layersNumber, 
                                                                                                                     layerSize, str(layerName).split('.')[-1].split("'")[0], 
                                                                                                                     dropout, loss, optimizer, bidirectional, epochs, batchSize)

    if (not os.path.exists(modelFilePath)):        
        # Create and train the Deep Learning model
        model = constructDLModel(featureColumns=featureColumns, numOfPastDays=numOfPastDays, numOfFutureDays=numOfFutureDays, layersNumber=layersNumber, layerSize=layerSize, layerName=layerName, 
                                 dropout=dropout, loss=loss, optimizer=optimizer, bidirectional=bidirectional)

        model.fit(x=processedData["XTrain"], y=processedData["YTrain"], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(processedData["XTest"], processedData["YTest"]))
        
        # Save the model
        model.save(modelFilePath)
        print("Trained model saved to {}".format(modelFilePath))
    else:
        # Load the model
        model = models.load_model(modelFilePath)
        print("Trained model loaded from {}".format(modelFilePath))
        
    # Testing
    #   Get actual and predicted data
    yActualData = processedData["YTest"]
    yPredictedData = model.predict(processedData["XTest"])

    #   Descale
    if (IS_SCALED_DATA):
        yActualData = np.squeeze(processedData["ColumnScalers"]["Close"].inverse_transform(yActualData))
        yPredictedData = np.squeeze(processedData["ColumnScalers"]["Close"].inverse_transform(yPredictedData))
   
    # print("================ Y-ACTUAL DATA ===================")
    # print(yActualData)
    # print("=============== Y-PREDICTED DATA =================")   
    # print(yPredictedData)

    #   When doing multi-step predicting, most of the future predicted days have more than 1 value. 
    #   E.g. "k" = 2; the original data: [..., a, b, c, d, ...]; using data before "a", "b" to get the predicted data of [b1, c1], [c2, d2] respectively; 
    #   so there is two predicted values for the data corresponding with the "c" values in the original data
    #   I reshape the "yActualData" and "yPredictedData" to become a 1D arrays, with one value only for each day
    (finalYActualData, finalYPredictedData) = ([], [])
    #     Reshape "yActualData"
    for i in range(len(yActualData)):
        if (i == 0):
            for j in range(0, numOfFutureDays):
                finalYActualData.append(yActualData[i][j])
        else:
            finalYActualData.append(yActualData[i][numOfFutureDays - 1])
    
    finalYActualData = np.array(finalYActualData)

    #     Reshape "yPredictedData: E.g. Given the data:
    #           [a0, b0, c0, d0]
    #           [b1, c1, d1, e0]
    #           [c2, d2, e1, f0]
    #           [d3, e2, f1, g0]
    #           [e3, f2, g1, h0]

    #       Append the values: a0; (b0 + b1) / 2; (c0 + c1 + c2) / 3
    i = 0
    while (i < numOfFutureDays - 1):
        sumValue = 0
        j = i
        while (j >= 0):
            sumValue += yPredictedData[i - j][j]
            j -= 1
            
        finalYPredictedData.append(sumValue / (i + 1))
        
        i += 1
    
    #       Append the values: (d0 + d1 + d2 + d3) / 4; (e0 + e1 + e2 + e3) / 4; 
    for i in range(len(yPredictedData) - numOfFutureDays + 1):
        sumValue = 0
        j = numOfFutureDays - 1
        while (j >= 0):
            sumValue += yPredictedData[i + (numOfFutureDays - j - 1)][j]
            j -= 1
        
        finalYPredictedData.append(sumValue / numOfFutureDays)
        
    #       Append the values: (f0 + f1 + f2) / 3; (g0 + g1) / 2; h0
    i = numOfFutureDays - 1
    while i > 0:
        sumValue = 0
        j = i
        while (j > 0):
            sumValue += yPredictedData[len(yPredictedData) - j][numOfFutureDays - i - 1 + j]
            j -= 1

        finalYPredictedData.append(sumValue / i)
            
        i -= 1
        
    finalYPredictedData = np.array(finalYPredictedData)
        
    # print("============== FINAL Y-ACTUAL DATA ===============")
    # print(finalYActualData)
    # print("============= FINAL Y-PREDICTED DATA =============")   
    # print(finalYPredictedData)
    
    #   Plot
    # plotSingleFeature("{} Close Price Multivariate and Multistep Prediction".format(COMPANY), finalYActualData, finalYPredictedData)
    
    return (finalYActualData, finalYPredictedData)

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST MACHINE LEARNING MODEL - ARIMA
def createARIMAModel(processedData):
    data = processedData["XTest"]
    actualData = processedData["YTest"]

    # Check if the model's folder is exist
    if (not os.path.isdir("models")):
        os.mkdir("models")     

    # Get the close data
    singleFeatureData = []
    for i in range(0, len(data)):
        singleFeatureSequence = []
        for sequence in data[i]:
            singleFeatureSequence.append(sequence[2])

        singleFeatureData.append(singleFeatureSequence)
        
    singleFeatureData = np.array(singleFeatureData)
        
    # Make sure the data become stationary
    singleFeatureStationaryData = []
    for singleFeatureSequence in singleFeatureData:
        singleFeatureStationaryData.append(makeStationaryData(data=singleFeatureSequence))
        
    singleFeatureStationaryData = np.array(singleFeatureStationaryData)

    predictedData = []

    # Create and fit ARIMA models
    for i in range(0, len(singleFeatureStationaryData)):
        modelFilePath = "models/{}-type_{}-company_{}-feature_{}-traingDataShape_{}-index.pkl".format("ARIMA", COMPANY, "Close", data.shape, i)        

        if (not os.path.exists(modelFilePath)):
            # Find the best parameters for the model for each sequence
            model = auto_arima(singleFeatureStationaryData[i], start_p=0, start_q=0, test="adf", max_p=7, max_q=7,
                               m=1, d=None, seasonal=False, start_P=0, D=0, trace=True,
                               error_action="ignore", suppress_warnings=True, stepwise=True)
        
            bestOrder = model.order
            bestSeasonalOrder = model.seasonal_order
        
            # Create model for each sequence
            model = ARIMA(singleFeatureStationaryData[i], order=bestOrder, seasonal_order=bestSeasonalOrder)
            # Fit the model
            fittedModel = model.fit()
            # Save the model
            fittedModel.save(modelFilePath)
            print("ARIMA model saved.")
        else:
            # Load the model
            fittedModel = ARIMAResults.load(modelFilePath)
            print("ARIMA model loaded.")  
        
        # Predict the data
        (forecastData, standardErrors, confidenceIntervals) = fittedModel.forecast(3, alpha=0.05)
        
        predictedData.append(forecastData)
       
    predictedData = np.array(predictedData)
    
    # Descale data
    actualData = np.squeeze(processedData["ColumnScalers"]["Close"].inverse_transform(actualData))
    predictedData = np.squeeze(processedData["ColumnScalers"]["Close"].inverse_transform(np.expand_dims(predictedData, axis=0)))
    
    print("=================== ACTUAL DATA ==================")
    print(actualData)
    print("================= PREDICTED DATA =================")
    print(predictedData)
    
    # Plot data
    # plotSingleFeature("{} Close Price Prediction with ARIMA model".format(COMPANY), actualData, predictedData)

    return (actualData, predictedData)
    
def makeStationaryData(data):
    stationaryData = data    

    aDFResult = adfuller(data, autolag="AIC")
    
    if (aDFResult[1] >= 0.05):
        dataLog = np.log(data)
        dataLog = pd.DataFrame(dataLog)
        movingAverage = dataLog.rolling(window=12).mean()
        dataLogMinusMean = dataLog - movingAverage
        dataLogMinusMean.ffill(inplace=True) 
        
    return stationaryData
    

processedData = processData(isStoredDataLocally=True, company=COMPANY, startDate=START_DATE, endDate=END_DATE, dataSource=DATA_SOURCE, 
                            numOfPastDays=NUMBER_OF_PAST_DAYS, numOfFutureDays=NUMBER_OF_FUTURE_DAYS, lookupSteps=LOOKUP_STEPS, featureColumns=FEATURE_COLUMNS,
                            trainRatio=TRAIN_RATIO, randomSplit=False, randomSeed=None, isScaledData=IS_SCALED_DATA, featureRange=(0, 1), isStoredScaler=True)

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST MACHINE LEARNING MODEL - ARIMA AND LSTM/GRU/RNN
# (secondYActualData, secondYPredictedData) = createARIMAModel(processedData)

# (firstYActualData, firstYPredictedData) = trainAndTestMultivariateDLModel(processedData=processedData, featureColumns=FEATURE_COLUMNS, 
#                                                                           numOfPastDays=NUMBER_OF_PAST_DAYS, numOfFutureDays=NUMBER_OF_FUTURE_DAYS, 
#                                                                           layersNumber=LAYERS_NUMBER, layerSize=LAYER_SIZE, layerName=LAYER_NAME, 
#                                                                           dropout=DROPOUT, loss=LOSS, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL,
#                                                                           epochs=EPOCHS, batchSize=BATCH_SIZE)

# (finalYActualData, finalYPredictedData) = ([], [])
# for i in range(0, min(len(firstYPredictedData), len(secondYPredictedData))):
#     finalYActualData.append((firstYActualData[i] + secondYActualData[i]) / 2)
#     finalYPredictedData.append((firstYPredictedData[i] + secondYPredictedData[i]) / 2)
    
#   Plot data
# plotSingleFeature("{} Close Price Prediction with ARIMA and LSTM/GRU/RNN model".format(COMPANY), finalYActualData, finalYPredictedData)

# trainAndTestMultistepDLModel(processedData=processedData, featureColumns=FEATURE_COLUMNS, numOfPastDays=NUMBER_OF_PAST_DAYS, numOfFutureDays=NUMBER_OF_FUTURE_DAYS, layersNumber=LAYERS_NUMBER, layerSize=LAYER_SIZE, layerName=LAYER_NAME, 
#                              dropout=DROPOUT, loss=LOSS, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL,
#                              epochs=EPOCHS, batchSize=BATCH_SIZE)

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST MACHINE LEARNING MODEL - RANDOM FOREST
def createRTModel(processedData, trainRatio):
    # Get the original data downloaded from Yahoo Finance
    data = processedData["Data"]
    
    # Split data
    trainSamples = int(trainRatio * len(data))
    (trainData, testData) = (data[:trainSamples], data[trainSamples:])
    
    # Get the xTrain, yTrain, xTest and yTest data
    xTrain = trainData[FEATURE_COLUMNS]
    yTrain = trainData["Close"]
    
    xTest = testData[FEATURE_COLUMNS]
    yTest = testData["Close"]
    
    modelFilePath = "models/{}-type_{}-company_{}-feature_{}-xTrainShape.pkl".format("RandomForest", COMPANY, "Close", xTrain.shape)   
    
    if (not os.path.exists(modelFilePath)):
        # Number of trees in Random Forest
        numOfEstimators = [500, 1000, 2000]
        # Maximum depth of levels
        maxDepth = [10, 20, 50]
        # Minimum number of samples required for node splitting
        minSamplesSplitting = [50, 100, 200]
        # Minimum number of samples required at each node (leaf)
        minSamplesEachNode = [2, 4, 8]
    
        testAccurateData = pd.DataFrame(columns=["NumOfEstimators", "MaxDepth", "MinSamplesSplitting", "MinSamplesEachNode", "AccuracyTrain", "AccuracyTest"])
    
        for element in list(itertools.product(numOfEstimators, maxDepth, minSamplesSplitting, minSamplesEachNode)):
            # Create the model
            model = RandomForestRegressor(n_estimators=element[0], max_depth=element[1], min_samples_split=element[2], min_samples_leaf=element[3])
            # Fit the model
            model.fit(xTrain, yTrain)
        
            # Train the model
            #   Predict with RF prediction method
            predictedTrainData = model.predict(xTrain)
            #   Compute the absolute errors
            errorsTrain = abs(predictedTrainData - yTrain)
            #   Compute the mean absolute error (in precentage)
            mapeTrain = 100 * (errorsTrain / yTrain)
            #   Compute the accuracy
            accuracyTrain = 100 - np.mean(mapeTrain)
        
            # Test data
            #   Predict with RF prediction method
            predictedTestData = model.predict(xTest)
            #   Compute the absolute errors
            errorsTest = abs(predictedTestData - yTest)
            #   Compute the mean absolute error (in precentage)
            mapeTest = 100 * (errorsTest / yTest)
            #   Compute the accuracy
            accuracyTest = 100 - np.mean(mapeTest)	
        
            testAccurateDataElement = pd.DataFrame(index = range(1), columns = ["NumOfEstimators", "MaxDepth", "MinSamplesSplitting", "MinSamplesEachNode", "AccuracyTrain", "AccuracyTest"])
        
            testAccurateDataElement.loc[:, "NumOfEstimators"] = element[0]
            testAccurateDataElement.loc[:, "MaxDepth"] = element[1]
            testAccurateDataElement.loc[:, "MinSamplesSplitting"] = element[2]
            testAccurateDataElement.loc[:, "MinSamplesEachNode"] = element[3]
            testAccurateDataElement.loc[:, "AccuracyTrain"] = accuracyTrain
            testAccurateDataElement.loc[:, "AccuracyTest"] = accuracyTest
        
            testAccurateData = pd.concat([testAccurateData, testAccurateDataElement], ignore_index=True)
        
        bestParameters = testAccurateData.loc[testAccurateData["AccuracyTest"] == max(testAccurateData["AccuracyTest"])]
        
        bestParameters = [bestParameters["NumOfEstimators"].values[0], bestParameters["MaxDepth"].values[0], bestParameters["MinSamplesSplitting"].values[0], bestParameters["MinSamplesEachNode"].values[0], 
                          bestParameters["AccuracyTrain"].values[0], bestParameters["AccuracyTest"].values[0]]
        
        print("============== TEST ACCURATE DATA ================")
        print(testAccurateData)
        print("================ BEST PARAMETERS =================")
        print(bestParameters)

        # Create and fit the model with the best found parameters
        model = RandomForestRegressor(n_estimators=bestParameters[0], max_depth=bestParameters[1], min_samples_split=bestParameters[2], min_samples_leaf=bestParameters[3])
        model.fit(xTrain, yTrain)
        
        # Save the model
        joblib.dump(model, modelFilePath)
    else:
        model = joblib.load(modelFilePath)
    
    # Predict the model
    yPredictedData = model.predict(xTest)

    yActualData = np.array(yTest)
    yPredictedData = np.array(yPredictedData)
    
    # Plot data
    # plotSingleFeature("{} Close Price Prediction with Random Forest Regressor".format(COMPANY), yActualData, yPredictedData)
    
    return (yActualData, yPredictedData)
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST MACHINE LEARNING MODEL - RANDOM FOREST AND LSTM/GRU/RNN
# (secondYActualData, secondYPredictedData) = createRTModel(processedData=processedData, trainRatio=TRAIN_RATIO)

# (firstYActualData, firstYPredictedData) = trainAndTestMultivariateDLModel(processedData=processedData, featureColumns=FEATURE_COLUMNS, 
#                                                                           numOfPastDays=NUMBER_OF_PAST_DAYS, numOfFutureDays=NUMBER_OF_FUTURE_DAYS, 
#                                                                           layersNumber=LAYERS_NUMBER, layerSize=LAYER_SIZE, layerName=LAYER_NAME, 
#                                                                           dropout=DROPOUT, loss=LOSS, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL,
#                                                                           epochs=EPOCHS, batchSize=BATCH_SIZE)

# (finalYActualData, finalYPredictedData) = ([], [])
# for i in range(0, min(len(firstYPredictedData), len(secondYPredictedData))):
#     finalYActualData.append((firstYActualData[i] + secondYActualData[i]) / 2)
#     finalYPredictedData.append((firstYPredictedData[i] + secondYPredictedData[i]) / 2)
    
#   Plot data
# plotSingleFeature("{} Close Price Prediction with RTRegressor and LSTM/GRU/RNN model".format(COMPANY), finalYActualData, finalYPredictedData)