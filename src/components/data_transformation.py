import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logger
from src.utils import save_object


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = str(
        PROJECT_ROOT / "artifacts" / "preprocessor.pkl"
    )


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "reading_score",
                "writing_score",
            ]

            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    (
                        "numerical",
                        numerical_pipeline,
                        numerical_columns,
                    ),
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_columns,
                    ),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "math_score"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            preprocessor = self.get_data_transformer_object()

            logger.info("Fitting preprocessing on training data")

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            save_object(
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessor,
            )

            logger.info("Data transformation complete")

            return (
                X_train_transformed,
                y_train.to_numpy(),
                X_test_transformed,
                y_test.to_numpy(),
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logger.exception("Data transformation failed")
            raise CustomException(e, sys) from e