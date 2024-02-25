# DATASET CLEANING
"""
Created on Mon Nov  8 22:54:48 2023
@author: javier
"""
from functions.def_functions import *
from paths.paths import path_base,folder_df_data_clean,file_df_data_clean

def mod_dtset_clean(df_data,start_date,endin_date):
    print('\n')
    print(f'START MODUL mod_dtset_clean')
    
    #Restart dataframe jic
    restart_dataframes = True  
    if 'df_data_clean' in locals() and restart_dataframes:del df_data_clean  # delete dataframe if exits 
            
    df_data_clean = df_data.copy()
    df_data_clean = add_index_column(df_data_clean)
    df_data_clean = date_anio(df_data_clean)
    df_data_clean = day_week(df_data_clean)
    df_data_clean = sort_columns(df_data_clean)
    df_data_clean = rounding_data(df_data_clean)
        
    # SAVE FILE with start_date and endin_date suffixes
    if not os.path.exists(os.path.join(path_base, folder_df_data_clean)):os.makedirs(os.path.join(path_base, folder_df_data_clean))
    file_df_data_clean = f"df_data_clean_{start_date}_{endin_date}.csv"
    save_file_path = os.path.join(path_base, folder_df_data_clean, file_df_data_clean)
    df_data_clean.to_csv(save_file_path, index=False)
    
    print(f'ENDIN MODUL mod_dtset_clean\n')
    return df_data_clean

if __name__ == "__main__":
    #Este bloque se ejecutará solo si el script se ejecuta directamente,
    #no cuando se importa como un módulo.
    mod_dtset_clean(df_data,start_date,endin_date)
