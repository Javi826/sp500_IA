#MODULE_PREPROCESSING
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""
from functions.def_functions import *
from paths.paths import path_base,folder_preprocessing

def mod_preprocessing (df_data_clean,filter_start_date,filter_endin_date):
    print(f'START MODUL mod_preprocessing')
    df_clean_filter = filter_data_by_date_range(df_data_clean, filter_start_date, filter_endin_date)
    selected_columns =['date','close','returns','direction','momentun','volatility','MA','day_week']
    df_preprocessing = pd.DataFrame(df_clean_filter, columns=selected_columns)
    
    df_preprocessing['returns'] = np.log(df_data_clean['close'] / df_data_clean['close'].shift(1))  
    df_preprocessing['direction'] = np.where(df_preprocessing['returns']>0, 1, 0) 
    df_preprocessing['momentun'] = df_preprocessing['returns'].rolling(5).mean().shift(1)
    df_preprocessing['volatility'] = df_preprocessing['returns'].rolling(20).std().shift(1)
    df_preprocessing['MA'] = df_preprocessing['close'].rolling(200).mean().shift(1)
    
    lags = 5  
    cols = []
    for lag in range(1,lags+1):
        col =f'lag_{lag}'
        df_preprocessing[col] = df_preprocessing['returns'].shift(lag)
        cols.append(col)
    df_preprocessing.dropna(inplace=True)
    
    # Print the first rows of the DataFrame with the assigned column names
    df_preprocessing['date'] = pd.to_datetime(df_preprocessing['date'])
    df_plots(df_preprocessing['date'],df_preprocessing['close'],'date','close','lines')
    
    # SAVE Dataframe
    excel_file_path = os.path.join(path_base, folder_preprocessing, "df_preprocessing.xlsx")
    df_preprocessing.to_excel(excel_file_path, index=False)
    
    print(f'ENDIN MODUL mod_preprocessing')
    print('\n')
    return df_preprocessing