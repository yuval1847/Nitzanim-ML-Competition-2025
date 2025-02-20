import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

class Model:
    """
    A class which represent the algorithem of the ML model for the Nitzanim ML Competition 2025.
    This model contains some Supervised Learning models which together creates Ensemble Learning.
    """

    # Constractor:
    def __init__(self, train_data:pd.core.frame.DataFrame, test_data:pd.core.frame.DataFrame):
        # The constractor of the Model

        # Variables which stores the data of the model (the train and test data) as DataFrame objects.
        self.train_data = train_data
        self.test_data = test_data

        # Removing the null values from the data
        self.Remove_Null_Values()

    
    # Preprocess functions:
    def Remove_Null_Values(self):
        # Input: Nothing.
        # Output: The function remove all the instances from the data which contains null values.
        self.train_data = self.train_data.dropna()
        self.test_data = self.test_data.dropna()


    # Visualization functions:
    def Generate_Histograms(self, quantitative_features:list[str]):
        # Input: A list of strings which represents a list of the quantitative features from the train data.
        # Output: The function shows histograms of the train data quantitative features.
        fig, axes = plt.subplots(len(quantitative_features), 1, figsize=(8, 20))
        for i, col in enumerate(quantitative_features):
            sns.histplot(self.train_data[col], bins=30, kde=True, ax=axes[i])
            axes[i].set_title(f"Histogram of {col}")
        plt.tight_layout()
        plt.show()
    def Generate_Bar_Charts(self, qualitative_features:list[str]):
        # Input: A list of strings which represents a list of the qualitative features from the train data.
        # Output: The function shows bar charts of the train data qualitative features.
        fig, axes = plt.subplots(len(qualitative_features), 1, figsize=(10, 40))
        for i, col in enumerate(qualitative_features):
            sns.countplot(y=self.train_data[col], order=self.train_data[col].value_counts().index, ax=axes[i], palette="viridis")
            axes[i].set_title(f"Bar Chart of {col}")
        plt.tight_layout()
        plt.show()
    def Generate_Scatter_Plots(self, quantitative_features:list[str]):
        # Input: A list of strings which represents a list of the quantitative features from the train data.
        # Output: The function shows scatter plots of the train data quantitative features.
        fig, axes = plt.subplots(len(quantitative_features), len(quantitative_features), figsize=(20, 20))
        for i, col_x in enumerate(quantitative_features):
            for j, col_y in enumerate(quantitative_features):
                if i != j:
                    sns.scatterplot(x=self.train_data[col_x], y=self.train_data[col_y], ax=axes[i, j], alpha=0.6)
                    axes[i, j].set_title(f"Scatter Plot of {col_x} vs. {col_y}")
                else:
                    axes[i, j].axis("off")
        plt.tight_layout()
        plt.show()
    def Visualization(self):
        # Input: Nothing.
        # Output: The function creates 3 types of visualization based on the train data,
        # which are histograms, bar charts and scatter plots.
        qualitative_features = ["WorkingCondition", "Education", "MaritalStatus", "Occupation", "Relationship", "Ethnicity", "Gender", "CountryOfOrigin"]
        quantitative_features = ["Age", "FinalWeight", "YearsOfEducation", "InvestmentGains", "InvestmentLosses", "WeeklyWorkHours"]
        self.Generate_Histograms(quantitative_features=quantitative_features)        
        self.Generate_Bar_Charts(qualitative_features=qualitative_features)
        self.Generate_Scatter_Plots(quantitative_features=quantitative_features) 
    