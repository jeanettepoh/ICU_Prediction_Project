import os
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass
from collections import Counter
from imblearn.over_sampling import SMOTENC

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    numerical_columns = ["age", "pulse", "sysbp"]
    categorical_columns = ["sex", "infection", "emergency"]
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.transformation_config = DataTransformationConfig()

    def get_data_transformation_obj(self):   
        try:
            numerical_pipeline = Pipeline(
                steps=[
                ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(drop='if_binary')),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logger.info(f"Numerical columns: {self.transformation_config.numerical_columns}")
            logger.info(f"Categorical columns: {self.transformation_config.categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("numerical_pipeline", numerical_pipeline, self.transformation_config.numerical_columns),
                    ("categorical_pipeline", categorical_pipeline, self.transformation_config.categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def address_target_class_imbalance(self, categorical_indices, X_train, y_train):
        try:
            old_counter = Counter(y_train)
            for target_class, count in old_counter.items():
                logger.info(f"No. of datapoints in target class {target_class}: {count} BEFORE applying SMOTENC")

            # Create the oversampler SMOTENC 
            smote_nc = SMOTENC(categorical_indices, random_state=42)
            X_train, y_train = smote_nc.fit_resample(X_train, y_train)

            new_counter = Counter(y_train)
            for target_class, count in new_counter.items():
                logger.info(f"No. of datapoints in target class {target_class}: {count} AFTER applying SMOTENC")

            return X_train, y_train
                        
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)
            logger.info("Read training and testing data completed")

            logger.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformation_obj()

            logger.info("Splitting input features and target")
            X_train = train_df.loc[:, train_df.columns != 'survive']
            y_train = train_df['survive']
            X_test = test_df.loc[:, test_df.columns != 'survive']
            y_test = train_df['survive']

            logger.info("Getting indices of categorical columns for SMOTENC")
            categorical_indices = [X_train.columns.get_loc(col) for col in self.transformation_config.categorical_columns]
            logger.info(f"The indices of categorical columns are {categorical_indices}")

            logger.info("Applying preprocessing object on training and testing data")
            X_train = preprocessing_obj.fit_transform(X_train)
            X_test = preprocessing_obj.transform(X_test)

            logger.info("Applying SMOTENC to address target class imbalance")
            X_train, y_train = self.address_target_class_imbalance(
                                    categorical_indices, 
                                    X_train, 
                                    y_train
                                )

            logger.info("Saving preprocessing object")
            save_object(
                file_path=self.transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            raise CustomException(e, sys)


# Check if above code works
if __name__ == "__main__":

    from src.components.data_ingestion import DataIngestion
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test = \
            data_transformation.initiate_data_transformation(train_data_path, test_data_path)