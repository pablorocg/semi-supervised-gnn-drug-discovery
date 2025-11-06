import os

from dotenv import load_dotenv

load_dotenv()


def get_data_dir() -> str:
    """
    Get the data directory from environment variable or use default.
    """
    source_data_dir = os.getenv("SOURCE_DATA_DIR", None)

    if source_data_dir is None:
        raise EnvironmentError("SOURCE_DATA_DIR environment variable is not set.")

    return source_data_dir


def get_splits_dir() -> str:
    """
    Get the splits directory from environment variable or use default.
    """
    splits_dir = os.getenv("SPLITS_DIR", None)

    if splits_dir is None:
        raise EnvironmentError("SPLITS_DIR environment variable is not set.")
    
    return splits_dir

def get_models_dir() -> str:
    """
    Get the models directory from environment variable or use default.
    """
    models_dir = os.getenv("MODELS_DIR", None)

    if models_dir is None:
        raise EnvironmentError("MODELS_DIR environment variable is not set.")
    
    return models_dir

def get_logs_dir() -> str:
    """
    Get the logs directory from environment variable or use default.
    """
    logs_dir = os.getenv("LOGS_DIR", None)

    if logs_dir is None:
        raise EnvironmentError("LOGS_DIR environment variable is not set.")
    
    return logs_dir

def ensure_dir_exists(dir_path: str) -> None:
    """
    Ensure that a directory exists; if not, create it.
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)




