#MODULE INIT
"""
Created on Sat Dec 30 21:06:19 2023
@author: jlahoz
"""

import yfinance as yf
import warnings
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

from statsmodels.tsa.arima.model import ARIMA

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, StandardScaler,MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

from tensorflow.keras import layers, models, initializers
import tensorflow as tf

# IGNORE WARNINGS
warnings.filterwarnings("ignore")

#VISUALIZATION PRINTS
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
