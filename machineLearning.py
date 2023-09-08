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
def trainDLModel(trainData, featureColumns=["Open", "High", "Low", "Close"], sequenceLength=50, layersNumber=2, layerSize=256, layerName=layers.LSTM, 
                     dropout=0.3, loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False,
                     epochs=50, batchSize=32):
    #   Check if the model's folder is exist
    if (not os.path.isdir("models")):
        os.mkdir("models")  
        
    modelFilePath = "models/{}-fCols_{}-pDays_{}-layers_{}-lSize_{}-lName_{}-do_{}-loss_{}-optimizer_{}biD_{}-epochs_{}-bSize.keras".format(len(featureColumns), sequenceLength, layersNumber, 
                                                                                                                     layerSize, str(layerName).split('.')[-1].split("'")[0], 
                                                                                                                     dropout, loss, optimizer, bidirectional, epochs, batchSize)

    if (not os.path.exists(modelFilePath)):
        # Extract the relevant features from trainData
        usedTrainData = trainData[0][:, :4]

        # print("================ USED TRAIN DATA =================")
        # print(usedTrainData)

        # Initialize empty lists for sequences and corresponding target values
        sequences = []
        targetValues = []

        # Create sequences and target values
        for i in range(len(usedTrainData) - PREDICTION_DAYS):
            sequence = usedTrainData[i:i + PREDICTION_DAYS] 
            targetValue = usedTrainData[i + PREDICTION_DAYS: i + PREDICTION_DAYS + 1]
    
            sequences.append(sequence)
            targetValues.append(targetValue)
    
        # Convert lists to NumPy arrays
        xTrain = np.array(sequences)
        yTrain = np.array(targetValues)

        # print("================ X-TRAIN DATA ====================")
        # print(xTrain)
        # print("================ Y-TRAIN DATA ====================")
        # print(yTrain)

        # Create and train the Deep Learning model
        model = constructDLModel(featureColumns=featureColumns, sequenceLength=sequenceLength, layersNumber=layersNumber, layerSize=layerSize, layerName=layerName, 
                                   dropout=dropout, loss=loss, optimizer=optimizer, bidirectional=bidirectional)

        model.fit(xTrain, yTrain, epochs=epochs, batch_size=batchSize)
        
        # Save the model
        model.save(modelFilePath)
        print("Trained model saved to {}".format(modelFilePath))
    else:
        # Load the model
        model = models.load_model(modelFilePath)
        print("Trained model loaded from {}".format(modelFilePath))
        
    return model

# Testing
def testDLModel(testData, testDates, model, featureColumns=["Open", "High", "Low", "Close"]):
    # Set up xTest data
    usedTestData = testData[0][:, :len(featureColumns)]
    
    sequences = []
    actualData = []
    predictedData = []
    
    for i in range(len(usedTestData) - PREDICTION_DAYS):
        sequence = usedTestData[i:i + PREDICTION_DAYS]
        actualValue = usedTestData[i + PREDICTION_DAYS:i + PREDICTION_DAYS + 1]
        
        # Predict the value using the model
        # predictedValue = model.predict(np.expand_dims(sequence, axis=0))

        sequences.append(sequence)
        actualData.append(actualValue)
        # predictedData.append(predictedValue)
        
    # Convert lists to NumPy arrays
    xTest = np.array(sequences)
    
    predictedData = model.predict(xTest)
    
    actualData = descaleAndConvertToDataFrame(data=actualData, testData=testData, featureColumns=featureColumns, testDates=testDates)
       
    print("================= ACTUAL DATA ====================")
    print(actualData)

    predictedData = descaleAndConvertToDataFrame(data=predictedData, testData=testData, featureColumns=featureColumns, testDates=testDates)
    
    print("================ PREDICTED DATA ==================")
    print(predictedData)

    return (actualData, predictedData)

def descaleAndConvertToDataFrame(data, testData, featureColumns, testDates):
    # Convert lists to NumPy arrays
    result = np.array(data)
    
    # Reshape actual and predicted data to 2D arrays
    result = result.reshape(-1, result.shape[-1])
    
    # Get the Scaler
    testScaler = testData[1]
    
    # The scaler is for the array of 6 columns, so I add (6 - len(featureColumns)) columns of "0" value to the right
    result = np.hstack((result, np.zeros((result.shape[0], (6 - len(featureColumns))))))

    # Descale the data
    result = testScaler.inverse_transform(result)
    
    # Get the dates for the actualData predictedData (except the first <len(featureColumns)> dates)
    dataDates = testDates[-(len(result)):]
    dataDates = pd.Series(dataDates, name="Date")

    # Convert to Pandas DataFrames, adding dates column and set indices
    result = pd.DataFrame(data=result, columns=featureColumns + ([None] * (6 - len(featureColumns))))
    result = pd.concat([dataDates, result], axis=1).set_index("Date")

    # Make sure the result is pandas.DataFrame (for correctly plotting)
    result = pd.DataFrame(data=result)
    
    return result

(trainData, testData, trainDates, testDates) = processData(True, COMPANY, START_DATE, END_DATE, DATA_SOURCE, 0.8, False, None, True, (0, 1), True)

model = trainDLModel(trainData=trainData, featureColumns=FEATURE_COLUMNS, sequenceLength=PREDICTION_DAYS, layersNumber=LAYERS_NUMBER, layerSize=LAYER_SIZE, layerName=LAYER_NAME, 
                              dropout=DROPOUT, loss=LOSS, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL,
                              epochs=EPOCHS, batchSize=BATCH_SIZE)

(actualData, predictedData) = testDLModel(testData, testDates, model, featureColumns=FEATURE_COLUMNS)  

# Plot each data in candlestick chart
# candlestickChartDisplay(actualData, 1)
# candlestickChartDisplay(predictedData, 1)

# Plot both in a line chart
plt.figure(figsize=(16, 9))

# Choose the feature column to plot. E.g. "Open"
plottingCol = "Close"

# Plot the actual data's Open price in blue
plt.plot(actualData.index, actualData[plottingCol], label="Actual {}".format(plottingCol), color='blue', linewidth=2)

# Plot the predicted data's Open price in orange
plt.plot(predictedData.index, predictedData[plottingCol], label="Predicted {}".format(plottingCol), color='orange', linestyle='--', linewidth=2)

# Set plot labels and title
plt.xlabel("Date")
plt.ylabel("Open Price")
plt.title("Actual vs. Predicted Open Price")

# Add a legend
plt.legend()

# Show the plot
plt.grid()
plt.show()

