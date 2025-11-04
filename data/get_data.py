
from pathlib import Path

from exact.utils.data import DataDownloader
from exact.core.values import DATASET_URL

DATA_DIR = Path(__file__).parent

if __name__ == "__main__":

    # Check if the data directory has any dirs inside

    if DataDownloader.has_subdirs(DATA_DIR):
        print("Data directory already exists and contains subdirectories.")
        print("Please remove the data directory or its contents to download the dataset again.")
        exit(1)
    else:
        print("Data directory is empty. Proceeding with download.")

        downloader = DataDownloader(dest_folder=str(DATA_DIR))
        downloader.download_dataset(DATASET_URL)

        print("Dataset downloaded and uncompressed successfully.")