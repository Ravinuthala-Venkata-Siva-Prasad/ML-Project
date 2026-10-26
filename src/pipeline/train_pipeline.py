
import sys

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException


def run_training_pipeline():
    try:
        print("Step 1: Starting Data Ingestion")

        train_path, test_path = (
            DataIngestion().initiate_data_ingestion()
        )

        print("Step 2: Starting Data Transformation")

        X_train, y_train, X_test, y_test, _ = (
            DataTransformation().initiate_data_transformation(
                train_path, test_path
            )
        )

        print("Step 3: Starting Model Training")

        results = ModelTrainer().initiate_model_trainer(
            X_train, y_train, X_test, y_test
        )

        print("Training completed successfully!")
        print("Results:", results)

        return results

    except Exception as e:
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    run_training_pipeline()
