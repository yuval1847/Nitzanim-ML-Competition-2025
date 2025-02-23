import pandas as pd
from Classes.Model import Model

# Reading the data in csv format
train_data = pd.read_csv("Data\\train_data.csv")
test_data = pd.read_csv("Data\\test_data.csv")

# Creating a Model object
model = Model(train_data=train_data,
              test_data=test_data)
# model.Visualization()
model.Label_Encoding()
model.Training()
print(f"The model evaluation:\n{model.Evaluation()}")
model.Testing()
model.Saving_Results()