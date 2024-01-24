#MODULE SUMMARY STATS

from paths.paths import *
from functions.def_functions import filter_data_by_date_range,df_plots,plots_histograms
import pandas as pd
from columns.columns import *

start_date = "1980-01-01"
endin_date = "2020-12-31"

# Construct the file name based on start_date and endin_date
file_df_data_clean = f"df_data_clean_{start_date}_{endin_date}.csv"
file_path_df_data_clean = os.path.join(path_base, folder_df_data_clean, file_df_data_clean)

# READING file
df_data_clean = pd.read_csv(file_path_df_data_clean, header=None, skiprows=1, names=columns_clean_order)

# STATS Range
filter_start_date = '2000-01-01'
filter_endin_date = '2019-12-31'

# filtering data by date
df_clean_filter = filter_data_by_date_range(df_data_clean, filter_start_date, filter_endin_date)

df_clean_filter['date'] = pd.to_datetime(df_clean_filter['date'])

#PLOT
df_plots(df_clean_filter['date'],df_clean_filter['close'],'date','close','lines')

# ANNUAL avg
mean_close_annual = df_clean_filter.groupby(df_clean_filter['date'].dt.year)['close'].mean()
print("Valor Medio Anual:")
print(mean_close_annual)

# ANNUAL MAX & MIN
max_close_annual = df_clean_filter.groupby(df_clean_filter['date'].dt.year)['close'].max()
min_close_annual = df_clean_filter.groupby(df_clean_filter['date'].dt.year)['close'].min()
print("Máximo Anual:")
print(max_close_annual)
print("Mínimo Anual:")
print(min_close_annual)

# ANNUAL Days +-
positive_days = df_clean_filter[df_clean_filter['close'] > df_clean_filter['open']].groupby(df_clean_filter['date'].dt.year)['date'].count()
negative_days = df_clean_filter[df_clean_filter['close'] < df_clean_filter['open']].groupby(df_clean_filter['date'].dt.year)['date'].count()
positive_ratio = positive_days / (positive_days + negative_days)
negative_ratio = negative_days / (positive_days + negative_days)
print("Proporción de Días Positivos:")
print(positive_ratio)
print("Proporción de Días Negativos:")
print(negative_ratio)

#ANNUAL profit
first_day_close = df_clean_filter.groupby(df_clean_filter['date'].dt.year)['close'].first()
last_day_close = df_clean_filter.groupby(df_clean_filter['date'].dt.year)['close'].last()
annual_returns_custom = (last_day_close - first_day_close) / first_day_close * 100
print("Rendimiento Anual (Método Personalizado):")
print(annual_returns_custom)

#ANNUAL volatility
volatility_annual = df_clean_filter.groupby(df_clean_filter['date'].dt.year)['var_day'].std() * (252 ** 0.5)
print("Volatilidad Anual:")
print(volatility_annual)

df_summary_stats = pd.DataFrame({
    'Year': mean_close_annual.index,  # Asumiendo que 'mean_close_annual' contiene los años
    'Mean_Close_Annual': mean_close_annual.values,
    'Max_Close_Annual': max_close_annual.values,
    'Min_Close_Annual': min_close_annual.values,
    'Positive_Ratio': positive_ratio.values,
    'Negative_Ratio': negative_ratio.values,
    '%_Annual': annual_returns_custom.values,
    'Volatility_Annual': volatility_annual.values
})

# SAVE Dataframe
excel_file_path = os.path.join(path_base, folder_summary_stats, f"df_summary_stats_{start_date}_{endin_date}.xlsx")
df_summary_stats.to_excel(excel_file_path, index=False)

#SUMARY
#



summary_stats_all = df_clean_filter.describe(include='all')
#print(summary_stats_all)
#HISTOGRAMS
#columns_of_interest = ['day_week','close','open','high', 'low','adj_close','var_day', 'volume']
#plots_histograms(df_clean_filter, columns_of_interest)
#PEARSONS
# numeric columns select
#columns_numeric = df_clean_filter.select_dtypes(include=['float', 'int']).columns
# Pearson + Sort ascdending
#pearsons = df_clean_filter[columns_numeric].corrwith(df_data_clean['close'], method='pearson').sort_values(ascending=False)
#print(pearsons)