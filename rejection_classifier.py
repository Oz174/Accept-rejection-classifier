import re 
import numpy as np 
import pandas as pd 
import openpyxl
import pickle 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report

class rejection_classifier():
    def __init__(self, model_file, vectorizer_file):
        with open(model_file,'rb') as model_file, open(vectorizer_file, 'rb') as vectorizer_file:
            self.clf = pickle.load(model_file)
            self.vectorizer = pickle.load(vectorizer_file)
            self.data = None
    
    def load_and_clean_data(self,data_file):
        df = pd.read_excel(data_file)
        self.preprocessed_data = df.copy()
        self.data = self.vectorizer.transform(df['Email'])
    

    # a function which outputs 0 or 1 based on our model
    def show_predictions_only(self):
        if (self.data is not None):
            pred_outputs = self.clf.predict(self.data)
            return pred_outputs
    
    # predict the outputs and the probabilities and 
    # add columns with these values at the end of the new data
    def show_results(self):
        if (self.data is not None):
            self.preprocessed_data['Status'] = self.clf.predict(self.data)
            return self.preprocessed_data
        
    def predict_and_save_to_excel(self):
        if (self.data is not None):
            self.preprocessed_data['Status'] = self.clf.predict(self.data)
            self.preprocessed_data['Status'] = self.preprocessed_data['Status'].apply(lambda x: 'Rejected' if x==0 else 'Accepted')
        self.preprocessed_data.to_excel('Result_All.xlsx',index=False)
        return
    
    def filter_accepted(self):
        if (self.data is not None):
            self.preprocessed_data['Status'] = self.clf.predict(self.data)
            self.preprocessed_data['Status'] = self.preprocessed_data['Status'].apply(lambda x: 'Rejected' if x==0 else 'Accepted')
        self.preprocessed_data = self.preprocessed_data[self.preprocessed_data['Status']=='Accepted']
        self.preprocessed_data.to_excel('Result_Accepted.xlsx',index=False)
        return
        