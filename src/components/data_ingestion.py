import os
import sys
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logger


@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join("artifacts", "raw_data.csv")
    train_data_path: str=os.path.join("artifacts", "train_data.csv")
    test_data_path: str=os.path.join("artifacts", "test_data.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Data Ingestion started")
        try:
            df = pd.read_csv(self.ingestion_config.raw_data_path)
            logger.info("Passed the raw data as dataframe")

            # Column names: Remove white spaces and convert to lower case
            df.columns = df.columns.str.strip().str.lower()

            # Taregt classes: 0 denoted Survived and 1 denotes Died
            df['survive'] = df['survive'].replace([0, 1], [1, 0])

            # Remove unnecessary columns
            df.drop(columns=['unnamed: 0', 'id', 'agegroup'], inplace=True)

            logger.info(f"The dataframe has dimensions {df.shape}")

            logger.info("Train test split initiated")

            # Generation of training and testing data
            train_df, test_df = train_test_split(
                df, test_size=0.2, stratify=df['survive'], random_state=24)

            train_df.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_df.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
 
            logger.info(f"The training set has dimensions {train_df.shape}")
            logger.info(f"The testing set has dimensions {test_df.shape}")

            logger.info("Data Ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# Check if above code works
if __name__ == "__main__":
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()