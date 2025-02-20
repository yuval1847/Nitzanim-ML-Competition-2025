import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from Classes.Model import Model

# Reading the data in csv format
train_data = pd.read_csv("Data\\train_data.csv")
test_data = pd.read_csv("Data\\test_data.csv")

# Creating a Model object
model = Model(train_data=train_data,
              test_data=test_data)
model.Visualization()