import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc

from dataProcessing import *

# ---------------------------------------------------------------------------------------------------------------------------------------------------

def preProcessDiplayedData(data, tradingDaysNumber):
    # Resample data to specified trading days (take the average value of <tradingDaysNumber> consecutive days)
    resampledData = data.resample("{}B".format(tradingDaysNumber)).agg({"Open": "mean", "High": "mean", "Low": "mean", "Close": "mean"})

    # Reset index to convert dates to numerical values for candlestick plotting
    resampledData.reset_index(inplace=True)
        
    # print("================ INITIAL DATA ====================")
    # print(data)
    # print("=============== RESAMPLED DATA ===================")
    # print(resampledData)
    
    return resampledData

def candlestickChartDisplay(data, tradingDaysNumber=1, title="Stock Candlestick Chart"):

    # Check if the number of trading days is less than 1
    if (tradingDaysNumber < 1):
        print("tradingDaysNumber must be equal or greater than 1.")
    else:
        resampledData = preProcessDiplayedData(data, tradingDaysNumber)  
        
        # Convert dates from a datetime format to numerical values (for plotting)
        resampledData["Date"] = mdates.date2num(resampledData["Date"].tolist())

        # Create a new figure and axes for the candlestick chart, specifying the initial size of the chart
        (fig, ax) = plt.subplots(figsize=(16, 9))
        
        # Plot the candlestick chart on these axes using the data from the DataFrame.
        candlestick_ohlc(ax, resampledData.values, width=0.6, colorup="g", colordown="r", alpha=0.8)

        # Format the date
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        fig.autofmt_xdate()

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(["Open", "High", "Low", "Close"])

        # Set the rotation angle for the x-axis (dates)
        plt.xticks(rotation=45)
        
        # Adjust the layout of the plot to prevent elements from overlapping and improve overall aesthetics
        plt.tight_layout()
        
        plt.show()
        

def boxplotChartDisplay(data, tradingDaysNumber=1, title="Stock Boxplot Chart"):

    # Check if the number of trading days is less than 1
    if (tradingDaysNumber < 1):
        print("tradingDaysNumber must be equal or greater than 1.")
    else:
        resampledData = preProcessDiplayedData(data, tradingDaysNumber)
        
        # print("============= PRETRANSPOSED DATA =================")
        # print(resampledData)
        
        # Transpose DataFrame so that columns become rows and rows become columns, as the output of the method ax.boxplot for resampledData 
        # seems not to be like the requirement - the x-axis would be Open/High/Low/Close instead of Date
        #   - Set 'Date' column as index and transpose the DataFrame
        transposedData = resampledData.set_index("Date").transpose()

        # print("=============== TRANSPOSED DATA ==================")
        # print(transposedData)

        # Create a new figure and axes for the boxplot chart, specifying the size of the figure
        (fig, ax) = plt.subplots(figsize=(16, 9))
        ax.boxplot(transposedData)
      
        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend(["Open", "High", "Low", "Close"])
        
        # Get the "Date" values from the DataFrame index
        dates = transposedData.columns
        # Format the dates to %Y-%m-%d
        dates = [date.strftime("%Y-%m-%d") for date in dates]
        # Set the "Date" values as x-axis ticks and rotate them
        plt.xticks(range(1, len(dates) + 1), dates, rotation=45)
        
        # Adjust the layout of the plot to prevent elements from overlapping and improve overall aesthetics
        plt.tight_layout()
        
        plt.show()
    
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# TEST DATA VISUALIZING

(trainData, testData) = processData(True, COMPANY, START_DATE, END_DATE, DATA_SOURCE, 0.8, False, None, False, (0, 1), False)

# candlestickChartDisplay(testData, 1)
# candlestickChartDisplay(testData, 3)

# boxplotChartDisplay(testData, 1)
# boxplotChartDisplay(testData, 2)
boxplotChartDisplay(testData, 5)
