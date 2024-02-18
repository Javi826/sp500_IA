# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 22:54:48 2024
@author: javier
"""
import os
columns_csv_yahoo= ['date','open','high','low','close','adj_close','volume']
columns_clean_order= ['index_id','date_anio','date', 'day_week', 'close', 'open', 'high', 'low', 'adj_close', 'returns','volume']
columns_preprocessing = ['pipe_returns', 'pipe_adj_close', 'pipe_open','pipe_high','pipe_low','pipe_volume','index_id','date_anio','date','day_week','close']
columns_preprocessing_order = ['index_id','date_anio','date', 'day_week', 'close', 'pipe_open', 'pipe_high', 'pipe_low', 'pipe_adj_close', 'pipe_returns', 'pipe_volume']