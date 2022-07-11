import pandas as pd
import numpy as np

def load_streetlight(filename):
    """
    Load Streetlight .csv data into a Pandas DataFrame.
    :param filename: Streetlight .csv filename
    :return: Pandas DataFrame
    """
    df = pd.read_csv(filename,
                usecols=['Origin Zone Name', 'Destination Zone Name',
                         'Origin Zone ID', 'Destination Zone ID',
                         'Average Daily O-D Traffic (StL Volume)', 'Avg Trip Length (mi)',
                         'Avg Trip Duration (sec)', 'Day Type', 'Day Part'])
    df = df[(df['Day Type'] == '1: Weekday (M-F)') & (df['Day Part'] != '00: All Day (12am-12am)')]
    return df
