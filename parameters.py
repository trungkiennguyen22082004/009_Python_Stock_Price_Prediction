from pyexpat import model
from keras import layers

DATA_SOURCE = "yahoo"
COMPANY = "TSLA"
COMPANY_NAME = "Tesla"

START_DATE = '2015-01-01'
END_DATE = '2022-12-31'

FEATURE_COLUMNS = ["Open", "High", "Low", "Close"]
IS_SCALED_DATA = True

TRAIN_RATIO = 0.8

# Number of days on which to base the prediction process
NUMBER_OF_PAST_DAYS = 60
# Number of future days predicted
NUMBER_OF_FUTURE_DAYS = 1

LOOKUP_STEPS = 10

LAYERS_NUMBER = 2
LAYER_SIZE = 64
LAYER_NAME = layers.GRU

LOSS = "mean_squared_error"
OPTIMIZER = "adam"
BATCH_SIZE = 32
EPOCHS = 100

DROPOUT = 0.3
BIDIRECTIONAL = False