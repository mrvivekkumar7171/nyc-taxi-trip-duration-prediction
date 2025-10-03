import pandas as pd
import pathlib
import logging
from sklearn.model_selection import train_test_split
from feature_definitions import feature_build


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_data(data_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    logging.info(f"Loading data from {data_path}")
    return pd.read_csv(data_path)


def save_data(train: pd.DataFrame, test: pd.DataFrame, output_path: str) -> None:
    """
    Save train and test datasets to the specified output path.

    Args:
        train (pd.DataFrame): Transformed training data.
        test (pd.DataFrame): Transformed test data.
        output_path (str): Output directory for processed data.
    """
    output_dir = pathlib.Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)

    logging.info(f"Processed train data saved to {train_path}")
    logging.info(f"Processed test data saved to {test_path}")


def process_and_save(train_path: str, test_path: str, output_path: str) -> None:
    """
    Load raw datasets, transform features, and save processed versions.

    Args:
        train_path (str): Path to raw train dataset.
        test_path (str): Path to raw test dataset.
        output_path (str): Path to save the processed datasets.
    """
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    train_features = feature_build(train_data, mode="train")
    test_features = feature_build(test_data, mode="test")

    save_data(train_features, test_features, output_path)


if __name__ == "__main__":
    curr_dir = pathlib.Path(__file__).resolve()
    home_dir = curr_dir.parent.parent.parent

    raw_data_dir = home_dir / "data" / "raw"
    processed_data_dir = home_dir / "data" / "processed"

    train_path = raw_data_dir / "train.csv"
    test_path = raw_data_dir / "test.csv"

    process_and_save(train_path.as_posix(), test_path.as_posix(), processed_data_dir.as_posix())
