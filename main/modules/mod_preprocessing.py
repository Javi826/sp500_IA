#MODULE_PREPROCESSING

from functions.def_functions import *
from paths.paths import *

def mod_preprocessing (df_data_clean,filter_start_date,filter_endin_date):
    print(f'START MODUL mod_preprocessing')
    df_clean_filter = filter_data_by_date_range(df_data_clean, filter_start_date, filter_endin_date)
    
    #PREPROCESSING
    null_imputer = SimpleImputer(strategy="constant", fill_value=None)
    
    raw_pipeline = make_pipeline(
        null_imputer,
        FunctionTransformer(),
        StandardScaler()
        #MinMaxScaler()
    )
    
    log_pipeline = make_pipeline(
        null_imputer,
        FunctionTransformer(np.log),
        StandardScaler()
        #MinMaxScaler()    
    )
    
    preprocessing = ColumnTransformer([
            ("raw", raw_pipeline, ['var_day','adj_close','open','high','low','volume'])
            #("log", log_pipeline, ['adj_close','open', 'high', 'low', 'volume'])
        ],
        remainder="passthrough")
    
    #X_preprocessing
    X_preprocessing = preprocessing.fit_transform(df_clean_filter)
      
    # Create a DataFrame with the transformed data and column names and ordered
    df_preprocessing = pd.DataFrame(X_preprocessing, columns=columns_preprocessing)
    df_preprocessing = df_preprocessing[columns_preprocessing_order]
    
    # Print the first rows of the DataFrame with the assigned column names
    df_preprocessing['date'] = pd.to_datetime(df_preprocessing['date'])
    df_plots(df_preprocessing['date'],df_preprocessing['close'],'date','close','lines')
    
    # SAVE Dataframe
    excel_file_path = os.path.join(path_base, folder_preprocessing, "df_preprocessing.xlsx")
    df_preprocessing.to_excel(excel_file_path, index=False)
    
    print(f'ENDIN MODUL mod_preprocessing')
    print('\n')
    return df_preprocessing