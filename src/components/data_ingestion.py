import sys
from pathlib import Path
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DataIngestionConfig:
    train_data_path: str = str(
        PROJECT_ROOT / "artifacts" / "train.csv"
    )
    test_data_path: str = str(
        PROJECT_ROOT / "artifacts" / "test.csv"
    )
    raw_data_path: str = str(
        PROJECT_ROOT / "artifacts" / "data.csv"
    )


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            data_path = PROJECT_ROOT / "notebook" / "data" / "stud.csv"

            logger.info("Reading dataset: %s", data_path)

            if not data_path.exists():
                raise FileNotFoundError(
                    f"Dataset not found: {data_path}"
                )

            df = pd.read_csv(data_path)

            if df.empty:
                raise ValueError("The dataset is empty.")

            required_columns = {
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                "reading_score",
                "writing_score",
                "math_score",
            }

            missing = required_columns - set(df.columns)

            if missing:
                raise ValueError(
                    f"Missing required columns: {sorted(missing)}"
                )

            output_dir = PROJECT_ROOT / "artifacts"
            output_dir.mkdir(parents=True, exist_ok=True)

            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
            )

            train_df, test_df = train_test_split(
                df,
                test_size=0.2,
                random_state=42,
            )

            train_df.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
            )
            test_df.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
            )

            logger.info(
                "Data ingestion complete. Train rows: %d; test rows: %d",
                len(train_df),
                len(test_df),
            )

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            logger.exception("Data ingestion failed")
            raise CustomException(e, sys) from e