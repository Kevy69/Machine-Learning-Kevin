import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def train_val_test_split(x, y, split_size, rand_state):

    # Train val test
    df1_x_train, df1_x_test, df1_y_train, df1_y_test = train_test_split(
        x, y, test_size=split_size, random_state=rand_state
    )

    # N / (1 - N)
    val_split_size = split_size / (1 - split_size)

    df1_x_train, df1_x_val, df1_y_train, df1_y_val = train_test_split(df1_x_train, df1_y_train, test_size=val_split_size, random_state=rand_state)
    
    return (df1_x_train, df1_x_val, df1_y_train, df1_y_val)