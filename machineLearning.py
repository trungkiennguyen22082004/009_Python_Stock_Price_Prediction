import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import layers, models, metrics

from parameters import *
from dataProcessing import *
# from dataVisualizing import *

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
        
    print("================ Y-ACTUAL DATA ===================")
    print(yActualData)
    print("=============== Y-PREDICTED DATA =================")   
    print(yPredictedData)
  
    #   Plot
    plt.figure(figsize=(16, 9))
    plt.title("{} Close Price Multivariate Prediction".format(COMPANY))
    plt.plot(yActualData, label="Actual Prices", color="blue")
    plt.plot(yPredictedData, label="Predicted Prices", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

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
   
    print("================ Y-ACTUAL DATA ===================")
    print(yActualData)
    print("=============== Y-PREDICTED DATA =================")   
    print(yPredictedData)

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
        
    print("============== FINAL Y-ACTUAL DATA ===============")
    print(finalYActualData)
    print("============= FINAL Y-PREDICTED DATA =============")   
    print(finalYPredictedData)
    
    #   Plot
    plt.figure(figsize=(16, 9))
    plt.title("{} Close Price Multistep Prediction".format(COMPANY))
    plt.plot(finalYActualData, label="Actual Prices", color="blue")
    plt.plot(finalYPredictedData, label="Predicted Prices", color="orange")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

processedData = processData(isStoredDataLocally=True, company=COMPANY, startDate=START_DATE, endDate=END_DATE, dataSource=DATA_SOURCE, 
                            numOfPastDays=NUMBER_OF_PAST_DAYS, numOfFutureDays=NUMBER_OF_FUTURE_DAYS, lookupSteps=LOOKUP_STEPS, featureColumns=FEATURE_COLUMNS,
                            trainRatio=0.8, randomSplit=False, randomSeed=None, isScaledData=IS_SCALED_DATA, featureRange=(0, 1), isStoredScaler=True)

# trainAndTestMultivariateDLModel(processedData=processedData, featureColumns=FEATURE_COLUMNS, numOfPastDays=NUMBER_OF_PAST_DAYS, numOfFutureDays=NUMBER_OF_FUTURE_DAYS, layersNumber=LAYERS_NUMBER, layerSize=LAYER_SIZE, layerName=LAYER_NAME, 
#                                dropout=DROPOUT, loss=LOSS, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL,
#                                epochs=EPOCHS, batchSize=BATCH_SIZE)

trainAndTestMultistepDLModel(processedData=processedData, featureColumns=FEATURE_COLUMNS, numOfPastDays=NUMBER_OF_PAST_DAYS, numOfFutureDays=NUMBER_OF_FUTURE_DAYS, layersNumber=LAYERS_NUMBER, layerSize=LAYER_SIZE, layerName=LAYER_NAME, 
                             dropout=DROPOUT, loss=LOSS, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL,
                             epochs=EPOCHS, batchSize=BATCH_SIZE)