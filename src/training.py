import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from src.logger import get_logger
from src.custom_exception import CustomExeption

import mlflow
import mlflow.sklearn

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self,processed_data_path="artifacts/processed"):
        self.processed_data_path = processed_data_path
        self.model_dir = "artifacts/models"
        os.makedirs(self.model_dir,exist_ok=True)

        logger.info("Model training initialized...")


    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.processed_data_path,"X_train.pkl",))
            self.X_test = joblib.load(os.path.join(self.processed_data_path,"X_test.pkl"))
            self.y_train = joblib.load(os.path.join(self.processed_data_path,"y_train.pkl"))
            self.y_test = joblib.load(os.path.join(self.processed_data_path,"y_test.pkl"))

            logger.info("Data loaded for model..")
        except Exception as e:
            logger.error(f"Error wile loading data for model {e}")
            raise CustomExeption(f"Faild to load data",e)
        
    def train_model(self):
        try:
            self.model = GradientBoostingClassifier(n_estimators=100 , learning_rate=0.1 , max_depth=3 , random_state=42)
            self.model.fit(self.X_train,self.y_train)

            joblib.dump(self.model,os.path.join(self.model_dir, "model.pkl"))
            logger.info("Model saved successfully...")

        except Exception as e:
            logger.error(f"Error wile training the model {e}")
            raise CustomExeption(f"Faild to train the model",e)
        
    def evaluate_model(self):
        try:
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)[ : , 1] if len(self.y_test.unique())== 2 else None

            accuracy = accuracy_score(self.y_test,y_pred)
            precision = precision_score(self.y_test,y_pred, average='weighted')
            recall = recall_score(self.y_test,y_pred,average='weighted')
            f1 = f1_score(self.y_test,y_pred,average='weighted')

            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision",precision)
            mlflow.log_metric("recall",recall)
            mlflow.log_metric("f1 score",f1)

            logger.info(f"Accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1_score: {f1}")

            roc_auc = roc_auc_score(self.y_test,y_proba)

            mlflow.log_metric("roc auc",roc_auc)
            logger.info(f"ROC_AUC_Value: {roc_auc_score}")
            
        except Exception as e:
            logger.error(f"Error wile loading data for model {e}")
            raise CustomExeption(f"Faild to load data",e)
        
    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__=="__main__":
    with mlflow.start_run():
        trainer = ModelTraining()
        trainer.run()

