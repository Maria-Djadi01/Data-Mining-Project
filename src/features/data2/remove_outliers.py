import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 5)
plt.rcParams['figure.dpi'] = 100

#----------------------------------------------------------------#
# Load data #
#----------------------------------------------------------------#

df = pd.read_csv('../../../data/interim/temp_dataset_processed.csv')
df.drop(columns=['Unnamed: 0', "Unnamed: 0.1"], inplace=True)

# --------------------------------------------------------------#
# Plotting outliers
# --------------------------------------------------------------#
for column in df.columns:
    if column not in ['zcta', 'Start date', 'end date', 'time_period', 'population']:
        df[[column, 'zcta']].boxplot(by='zcta', figsize=(20, 10))
        plt.title(f'Boxplot for {column}')
        plt.show()
        


