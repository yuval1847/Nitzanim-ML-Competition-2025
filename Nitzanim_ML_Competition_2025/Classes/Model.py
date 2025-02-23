import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from Classes.EModels import EModels
from Classes.ETypeOfScaling import ETypeOfScaling


class Model:
    """
    A class which represent a ML model algorithem for the Nitzanim ML Competition 2025.
    This model contains some Supervised Learning models which together creates Ensemble Learning Model.
    """

    # Constractor:
    def __init__(self, train_data:pd.core.frame.DataFrame, test_data:pd.core.frame.DataFrame):
        # The constractor of the Model

        # Variables which stores the data of the model (the train and test data) as DataFrame objects.
        self.train_data = train_data
        self.test_data = test_data
        
        # 2 lists of the features' names divided by representing qualitative or quantitative data types.
        self.qualitative_features = ["WorkingCondition", "Education", "MaritalStatus", "Occupation", "Relationship", "Ethnicity", "Gender", "CountryOfOrigin", "MonthlyIncome"]
        self.quantitative_features = ["Age", "FinalWeight", "YearsOfEducation", "InvestmentGains", "InvestmentLosses", "WeeklyWorkHours"]
        
        # Removing the null values from the data
        self.Remove_Null_Values()

        # Normalized data
        self.normalized_train_data = self.Normalization(self.normalized_train_data)
        self.normalized_test_data = self.Normalization(self.normalized_test_data)

        # Standardized data
        self.standardized_train_data = self.Standardization(self.standardized_train_data)
        self.standardized_test_data =self.Standardization(self.standardized_test_data)

        # A dictionary of scaling types per model
        self.models_type_of_scaling = {
            EModels.LOGISTIC_REGRESSION:ETypeOfScaling.NO_SCALING,
            EModels.RANDOM_FOREST:ETypeOfScaling.NO_SCALING,
            EModels.KNN:ETypeOfScaling.STANDARDIZATION,
            EModels.GAUSSIAN_NAIVE_BAYES:ETypeOfScaling.NO_SCALING,
            EModels.SVM:ETypeOfScaling.STANDARDIZATION
        }

    
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
    def Generate_Box_Plots(self, quantitative_features:list[str]):
        # Input: A list of strings which represents a list of the quantitative features from the train data.
        # Output: The function shows box plots of the train data quantitative features.
        fig, axes = plt.subplots(len(quantitative_features), 1, figsize=(8, 20))
        for i, col in enumerate(quantitative_features):
            sns.boxplot(x=self.train_data[col], ax=axes[i], palette="coolwarm")
            axes[i].set_title(f"Box Plot of {col}")
        plt.tight_layout()
        plt.show()
    def Visualization(self):
        # Input: Nothing.
        # Output: The function creates 3 types of visualization based on the train data,
        # which are histograms, bar charts and scatter plots.
        self.Generate_Histograms(quantitative_features=self.quantitative_features)        
        self.Generate_Bar_Charts(qualitative_features=self.qualitative_features)
        self.Generate_Box_Plots(quantitative_features=self.quantitative_features) 
    

    # Label Encoding functions:
    def Label_Encoding_Specific_Data(self, data:pd.core.frame.DataFrame):
        # Input: DataFrame object which contains data.
        # Output: The given DataFrame object after performing label encoding over it.
        for i in self.qualitative_features:
            data = LabelEncoder.fit_transform(data[i])
        return data
    def Label_Encoding(self):
        # Input: Nothing.
        # Output: The function performs label encoding over the train and test data.
        self.train_data = self.Label_Encoding_Specific_Data(self.train_data)
        self.test_data = self.Label_Encoding_Specific_Data(self.test_data)


    # Normalization and Standardization functions:
    def Normalization(self, data:pd.core.frame.DataFrame):
        # Input: DataFrame object which contains data.
        # Output: The given DataFrame object after performing normalization over it.
        for i in self.quantitative_features:
            if i != "MonthlyIncome":
                data[i] = MinMaxScaler(feature_range=(0, 1)).fit_transform(data[[i]])
        return data
    def Standardization(self, data: pd.core.frame.DataFrame):
        # Input: DataFrame object which contains data.
        # Output: The given DataFrame object after performing standardization over it.
        for i in self.quantitative_features:
            if i != "MonthlyIncome":
                data[i] = StandardScaler().fit_transform(data[[i]])
        return data
    

    # Spliting data function:
    def Spliting_Data(self, data:pd.core.frame.DataFrame):
        # Input: DataFrame object which contains data.
        # Output: The splited parts of the given data (the data without the MonthlyIncome column
        # and the data with only the MonthlyIncome column).
        return data.drop('MonthlyIncome', axis=1), data['MonthlyIncome']


    # Supervised Models functions:
    def Logistic_Regression_Model(self, train_data_x:pd.core.frame.DataFrame, train_data_y:pd.core.frame.DataFrame):
        # Input: DataFrame objects which represent the 2 splited parts of the train data.
        # Output: The function returns a Logistic Regression model after training it.
        logistic_regression_model = LogisticRegression(max_iter=1000)
        logistic_regression_model.fit(train_data_x, train_data_y)
        return logistic_regression_model
    def Random_Forest_Model(self, train_data_x:pd.core.frame.DataFrame, train_data_y:pd.core.frame.DataFrame):
        # Input: DataFrame objects which represent the 2 splited parts of the train data.
        # Output: The function returns a Random Forest model after training it.
        random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
        random_forest_model.fit(train_data_x, train_data_y)
        return random_forest_model
    def KNN_Model(self, standardized_train_data_x:pd.core.frame.DataFrame, standardized_train_data_y:pd.core.frame.DataFrame):
        # Input: DataFrame objects which represent the 2 splited parts of the standardized train data.
        # Output: The function returns a KNN model after training it.
        knn_model = KNeighborsClassifier(n_neighbors=5)
        knn_model.fit(standardized_train_data_x, standardized_train_data_y)
        return knn_model
    def Gaussian_Naive_Bayes(self, train_data_x:pd.core.frame.DataFrame, train_data_y:pd.core.frame.DataFrame):
        # Input: DataFrame objects which represent the 2 splited parts of the train data.
        # Output: The function returns a Gaussian Naive Bayes model after training it.
        gaussian_naive_bayes = GaussianNB()
        gaussian_naive_bayes.fit(train_data_x, train_data_y)
        return gaussian_naive_bayes
    def SVM_Model(self, standardized_train_data_x:pd.core.frame.DataFrame, standardized_train_data_y:pd.core.frame.DataFrame):
        # Input: DataFrame objects which represent the 2 splited parts of the train data.
        # Output: The function returns a Support Vector Machine model after training it.
        svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
        svm_model.fit(standardized_train_data_x, standardized_train_data_y)
        return svm_model


    # Training function:
    def Training(self):
        # Input: Nothing.
        # Output: A dictionary of the the models after inserting them the training data.
        train_data_x, train_data_y = self.Spliting_Data(self.train_data)
        normalized_train_data_x, normalized_train_data_y = self.Spliting_Data(self.normalized_train_data)
        standardized_train_data_x, standardized_train_data_y = self.Spliting_Data(self.standardized_train_data)

        models = {
            EModels.LOGISTIC_REGRESSION:self.Logistic_Regression_Model(train_data_x, train_data_y),
            EModels.RANDOM_FOREST:self.Random_Forest_Model(train_data_x, train_data_y),
            EModels.KNN:self.KNN_Model(standardized_train_data_x, standardized_train_data_y),
            EModels.GAUSSIAN_NAIVE_BAYES:self.Gaussian_Naive_Bayes(train_data_x, train_data_y),
            EModels.SVM:self.SVM_Model(standardized_train_data_x, standardized_train_data_y)
        }

        return models


    # Testing function:
    def Testing(self, models:dict[EModels:pd.core.frame.DataFrame]):
        # Input: A dictionary of enums and the trained models.
        # Output: A dictionary of DataFrame objects which represent the predictions.
        test_data_x, test_data_y = self.Spliting_Data(self.test_data)
        normalized_test_data_x, normalized_test_data_y = self.Spliting_Data(self.normalized_test_data)
        standardized_test_data_x, standardized_test_data_y = self.Spliting_Data(self.standardized_test_data)
        
        predictions = {}

        for key, value in models:
            if self.models_type_of_scaling[key] == ETypeOfScaling.NO_SCALING:
                predictions[key] = value.predict(test_data_x)
            elif self.models_type_of_scaling[key] == ETypeOfScaling.NORMALIZATION:
                predictions[key] = value.predict(normalized_test_data_x)
            elif self.models_type_of_scaling[key] == ETypeOfScaling.STANDARDIZATION:
                predictions[key] = value.predict(standardized_test_data_x)
            else:
                raise Exception(f"No type of scaling was set to {key.name} model")
        
        return predictions
    

    # Evaluation functions:
    def Evaluation(self):
        # Input: A dictionary of enums and DataFrame objects which represent the predictions.
        # Output: A string which contains the models evaluations according to several evaluation methods.
        