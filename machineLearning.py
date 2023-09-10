import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras import layers, models

from parameters import *
from dataProcessing import *
# from dataVisualizing import *

# ---------------------------------------------------------------------------------------------------------------------------------------------------

def constructDLModel(featureColumns=["Open", "High", "Low", "Close"], sequenceLength=50, layersNumber=2, layerSize=256, layerName=layers.LSTM, 
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
                model.add(layers.Bidirectional(layerName(units=layerSize, return_sequences=True), batch_input_shape=(None, sequenceLength, len(featureColumns))))
            else:
                model.add(layerName(units=layerSize, return_sequences=True, batch_input_shape=(None, sequenceLength, len(featureColumns))))
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
    
    # Add Dense output layer (with <len(featureColumns)> neuron (unit) and a linear activation function)
    model.add(layers.Dense(len(featureColumns), activation="linear"))
    
    # Compile the model (with loss function, evaluation metric (mean_absolute_error), and optimizer)
    model.compile(loss=loss, metrics=["mean_absolute_error"], optimizer=optimizer)
    
    return model

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST MACHINE LEARNING MODEL

# Training
def trainDLModel(processedData, featureColumns=["Open", "High", "Low", "Close"], sequenceLength=50, layersNumber=2, layerSize=256, layerName=layers.LSTM, 
                     dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False,
                     epochs=50, batchSize=32):
    #   Check if the model's folder is exist
    if (not os.path.isdir("models")):
        os.mkdir("models")  
        
    modelFilePath = "models/{}-fCols_{}-pDays_{}-layers_{}-lSize_{}-lName_{}-do_{}-loss_{}-optimizer_{}biD_{}-epochs_{}-bSize.keras".format(len(featureColumns), sequenceLength, layersNumber, 
                                                                                                                     layerSize, str(layerName).split('.')[-1].split("'")[0], 
                                                                                                                     dropout, loss, optimizer, bidirectional, epochs, batchSize)

    if (not os.path.exists(modelFilePath)):        
        # Create and train the Deep Learning model
        model = constructDLModel(featureColumns=featureColumns, sequenceLength=sequenceLength, layersNumber=layersNumber, layerSize=layerSize, layerName=layerName, 
                                   dropout=dropout, loss=loss, optimizer=optimizer, bidirectional=bidirectional)

        model.fit(x=processedData["XTest"], y=processedData["YTest"], epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(processedData["XTest"], processedData["YTest"]))
        
        # Save the model
        model.save(modelFilePath)
        print("Trained model saved to {}".format(modelFilePath))
    else:
        # Load the model
        model = models.load_model(modelFilePath)
        print("Trained model loaded from {}".format(modelFilePath))
        
    return model

# Testing

processedData = processData(isStoredDataLocally=True, company=COMPANY, startDate=START_DATE, endDate=END_DATE, dataSource=DATA_SOURCE, 
                            predictionDays=PREDICTION_DAYS, lookupSteps=LOOKUP_STEPS, featureColumns=FEATURE_COLUMNS,
                            trainRatio=0.8, randomSplit=False, randomSeed=None, isScaledData=IS_SCALED_DATA, featureRange=(0, 1), isStoredScaler=True)

model = trainDLModel(processedData=processedData, featureColumns=FEATURE_COLUMNS, sequenceLength=PREDICTION_DAYS, layersNumber=LAYERS_NUMBER, layerSize=LAYER_SIZE, layerName=LAYER_NAME, 
                              dropout=DROPOUT, loss=LOSS, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL,
                              epochs=EPOCHS, batchSize=BATCH_SIZE)

# Get actual and predicted data
yActualData = processedData["YTest"]
yPredictedData = model.predict(processedData["XTest"])

# Descale
if (IS_SCALED_DATA):
    yActualData = np.squeeze(processedData["ColumnScalers"]["Close"].inverse_transform(np.expand_dims(yActualData, axis=0)))
    yPredictedData = np.squeeze(processedData["ColumnScalers"]["Close"].inverse_transform(yPredictedData))
    
# Get the Close predicted data
yPredictedData = [row[3] for row in yPredictedData]
   
print("================ Y-ACTUAL DATA ===================")
print(yActualData)
print("=============== Y-PREDICTED DATA =================")   
print(yPredictedData)

# Plot
plt.figure(figsize=(16, 9))
plt.title("{} Close Price Prediction".format(COMPANY))
plt.plot(yActualData, label="Actual Prices", color="blue")
plt.plot(yPredictedData, label="Predicted Prices", color="orange")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()