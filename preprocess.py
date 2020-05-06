# -*- coding: utf-8 -*-
"""

@author: Samip
"""

class PreProcess:
    
    def read_data(self):    
        #Importing dataset
        import pandas as pd
        df = pd.read_csv('iphone_purchase_records.csv')
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        return x, y
    
    def preprocessing(self):
        x, y = self.read_data()
        #Convert Gender to numbers
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        x[:, 0] = le.fit_transform(x[:, 0])
        return x, y
    
    def split_data(self):
        x, y = self.preprocessing()
        #Splitting the dataset to training and testing
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.75, random_state = 0)
        return x_train, x_test, y_train, y_test
    
    def scale_data(self):
        x_train, x_test, y_train, y_test = self.split_data()
        #Feature Scaling
        from sklearn.preprocessing import StandardScaler
        x_sc = StandardScaler()
        x_train = x_sc.fit_transform(x_train)
        x_test = x_sc.fit_transform(x_test)
        return x_train, x_test, y_train, y_test