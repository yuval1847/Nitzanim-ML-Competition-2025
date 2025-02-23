import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score


class Model:
    """
    A class which represent a ML model algorithem for the Nitzanim ML Competition 2025.
    This model contains some Supervised Learning models which together creates a Stacking Ensemble Learning Model.
    """

    # Constractor:
    def __init__(self, train_data:pd.DataFrame, test_data:pd.DataFrame):
        # The constractor of the Model

        # Variables which stores the data of the model (the train and test data) as DataFrame objects.
        self.train_data = train_data
        self.test_data = test_data
        
        # 2 lists of the features' names divided by representing qualitative or quantitative data types.
        self.qualitative_features = ["WorkingCondition", "Education", "MaritalStatus", "Occupation", "Relationship", "Ethnicity", "Gender", "CountryOfOrigin"]
        self.quantitative_features = ["Age", "FinalWeight", "YearsOfEducation", "InvestmentGains", "InvestmentLosses", "WeeklyWorkHours"]
        
        # A variable which stores the StackingClassifier object of the model itself.
        self.stacking_model = None

        # A variable which stores the y prediction of the train data.
        self.pred_y = pd.DataFrame()

        # A variable which stores the y prediction the test data.
        self.test_pred_y = pd.DataFrame()

        # Removing the null values from the data
        self.Remove_Null_Values()

    
    # Preprocess functions:
    def Remove_Null_Values(self):
        # Input: Nothing.
        # Output: The function remove all the instances from the data which contains null values.
        X_train = self.train_data.drop(['MonthlyIncome'], axis=1)
        y_train = self.train_data['MonthlyIncome']

        imputer = SimpleImputer(strategy='most_frequent')
        X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
        self.train_data = pd.concat([X_train, y_train], axis=1)
        self.test_data = pd.DataFrame(imputer.transform(self.test_data), columns=self.test_data.columns)



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
    

    # Label Encoding function:
    def Label_Encoding(self):
        # Input: Nothing.
        # Output: The function performs label encoding over the train and test data.
        for i in self.qualitative_features:
            le = LabelEncoder()
            self.train_data.loc[:, i] = le.fit_transform(self.train_data[i].astype(str))
            self.test_data.loc[:, i] = le.transform(self.test_data[i].astype(str))


    # Spliting data function:
    def Spliting_Train_Data(self):
        # Input: DataFrame object which contains data.
        # Output: The splited parts of the given data (the data without the MonthlyIncome column
        # and the data with only the MonthlyIncome column).
        return train_test_split(self.train_data.drop(['Id', 'MonthlyIncome'], axis=1), self.train_data['MonthlyIncome'], test_size=0.2, random_state=42)
            

    # Training function:
    def Training(self):
        # Input: Nothing.
        # Output: The function trains the models and store it's y prediction.
        train_data_x, value_x, train_data_y, _ = self.Spliting_Train_Data()

        models = [
            ('dt', DecisionTreeClassifier()),
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('nb', GaussianNB()),
            ('knn', Pipeline([
                ('scaler', StandardScaler()), 
                ('knn', KNeighborsClassifier(n_neighbors=5))
            ])),
            ('svm', Pipeline([
                ('scaler', StandardScaler()), 
                ('svm', SVC(probability=True))
            ])),
            ('lr', Pipeline([
                ('scaler', StandardScaler()), 
                ('lr', LogisticRegression())
            ]))
        ]

        self.stacking_model = StackingClassifier(estimators=models, final_estimator=LogisticRegression())
        self.stacking_model.fit(train_data_x, train_data_y)
        self.pred_y = self.stacking_model.predict(value_x)
        print("Finished Training!")
        

    # Evaluation function:
    def Evaluation(self):
        # Input: Nothing.
        # Output: A string which contains the evaluation scores of the model.
        _, _, _, value_y = self.Spliting_Train_Data()
        return f"Accuracy: {accuracy_score(value_y, self.pred_y)}\nPrecision: {precision_score(value_y, self.pred_y)}\nRecall: {recall_score(value_y, self.pred_y)}"
    

    # Testing function:
    def Testing(self):
        # Input: Nothing.
        # Output: The function test the model.
        test_x = self.test_data.drop('Id', axis=1)
        self.test_pred_y = self.stacking_model.predict(test_x)
        print("Finished Testing!")
    

    # Saving results function:
    def Saving_Results(self):
        # Input: Nothing.
        # Output: The function creates a csv file named "results.csv" which stores the results of the model.
        pd.DataFrame({
            'Id': self.test_data['Id'],
            'MonthlyIncome': self.test_pred_y
        }).to_csv('Data\\results.csv', index=False)
        print("The results were saved successfully!")