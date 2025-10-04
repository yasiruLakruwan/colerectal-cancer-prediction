import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.feature_selection import chi2,SelectKBest

from logger import get_logger
from custom_exception import CustomExeption

logger = get_logger(__name__)

class DataProcessing:
    def __init__(self,input_path,output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.df = None
        self.X = None
        self.y = None
        self.scaled_features = []

        os.makedirs(output_path,exist_ok=True)
        logger.info("Data processing initialized....")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("CSV file loaded..")
        except Exception as e:
            logger.error(f"Error wile reading csv file {e}")
            raise CustomExeption(f"Error while csv file loading",e)
    
    def process_data(self):
        try:
            self.df = self.df.drop(columns=['Patient_ID'])
            self.X = self.df.drop(columns=['Survival_Prediction'])
            self.y = self.df["Survival_Prediction"]

            categorical_cols = self.X.select_dtypes(include=['object']).columns
            label_encoders = {}

            for col in categorical_cols:
                le = LabelEncoder()
                self.X[col] = le.fit_transform(self.X[col])
                self.label_encoders[col] = le 
            logger.info("Basic processing done...")
        
        except Exception as e:
            logger.error(f"Error wile preprocess data {e}")
            raise CustomExeption(f"Faild to process data",e)
    
    def feature_selection(self):
        try:
            X_train , _ , y_train , _ = train_test_split(self.X,self.y , test_size=0.2 , random_state=42)
            ### CHI-SQUARe-TEST

            X_cat = X_train.select_dtypes(include=['int64' , 'float64'])
            chi2_selector = SelectKBest(score_func=chi2 , k="all")
            chi2_selector.fit(X_cat,y_train)

            chi2_scores = pd.DataFrame({
                    'Feature' : X_cat.columns,
                    "Chi2 Score" : chi2_selector.scores_
                }).sort_values(by='Chi2 Score' , ascending=False)
            
            top_features = chi2_scores.head(5)["Feature"].tolist()
            self.scaled_features = top_features
            logger.info(f"Selected features are:{self.scaled_features}")

            self.X = self.X[self.scaled_features]
            logger.info("Feature selection done...")
        
        except Exception as e:
            logger.error(f"Error wile feature selection {e}")
            raise CustomExeption(f"Faild to select features",e)
        
    def split_scale_data(self):
        try:
            X_train , X_test , y_train , y_test = train_test_split(self.X,self.y , test_size=0.2 , random_state=42 , stratify=self.y)

            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)

            logger.info("Spliting and scaling done..")

            return X_train,X_test,y_train,y_test
        
        except Exception as e:
            logger.error(f"Error wile spliting and scaling data {e}")
            raise CustomExeption(f"Faild to split and scale data",e)
    
    def save_data_scaler(self,X_train,X_test,y_train,y_test):
        try:
            joblib.dump(X_train,os.path.join(self.output_path,'X_train.pkl'))
            joblib.dump(X_test,os.path.join(self.output_path,'X_test.pkl'))
            joblib.dump(y_train,os.path.join(self.output_path,'y_train.pkl'))
            joblib.dump(y_test,os.path.join(self.output_path,'y_test.pkl'))

            joblib.dump(self.scaler,os.path.join(self.output_path,'scaler.pkl'))

            logger.info("All the saving parts are ")
        
        except Exception as e:
            logger.error(f"Error wile saving scaler and train test data {e}")
            raise CustomExeption(f"Faild to save the scaler and train test data",e)
    
    def run(self):
        self.load_data()
        self.process_data()
        self.feature_selection()
        X_train,X_test,y_train,y_test = self.split_scale_data()
        self.save_data_scaler( X_train,X_test,y_train,y_test)

        logger.info("Data processing pipeline executed successfully.")


if __name__ == "__main__":
    input_path = "artifacts/raw/data.csv"
    output_path = "artifacts/processed"

    processor = DataProcessing(input_path,output_path)
    processor.run()