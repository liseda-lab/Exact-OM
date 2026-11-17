import shutil
import tarfile
import zipfile
from pathlib import Path

import pandas as pd
import requests
import yaml
from tqdm import tqdm

# Read data tables

na_vals = pd.io.parsers.readers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})


def read_table(file_path: Path) -> pd.DataFrame:
    """Read tsv file as pandas dataframe without treating "null" as empty string."""

    file_path = str(file_path)

    sep = "\t" if file_path.endswith(".tsv") else ","
    return pd.read_csv(file_path, sep=sep, na_values=na_vals, keep_default_na=False)


# Save dictionaries to CSV


def save_dict_to_csv(data: dict, file_path: Path, sep: str = ",", columns: list = ["key", "value"]):
    """Save a dictionary to a csv file."""
    pd.DataFrame(data.items(), columns=columns).to_csv(str(file_path), index=False, sep=sep)


# Read YAML


def read_yaml(file_path: Path):
    with open(str(file_path), "r") as file:
        return yaml.safe_load(file)


# Download data from URLs


class DataDownloader:
    def __init__(self, dest_folder):
        self.dest_folder = Path(dest_folder)
        self.dest_folder.mkdir(parents=True, exist_ok=True)

    def download_file(self, url, filename=None, skip_existing=True):
        local_filename = self.dest_folder / (filename or url.split("/")[-1])
        if skip_existing and local_filename.exists():
            print(f"{local_filename.name} already present; skipping download.")
            return local_filename

        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024
        t = tqdm(total=total_size, unit="iB", unit_scale=True)
        with local_filename.open("wb") as f:
            for chunk in response.iter_content(chunk_size=block_size):
                t.update(len(chunk))
                f.write(chunk)
        t.close()
        return local_filename

    def unzip_file(self, zip_path, dest_folder=None):
        destination = Path(dest_folder) if dest_folder is not None else self.dest_folder
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(destination)

    def untar_file(self, tar_path, dest_folder=None):
        destination = Path(dest_folder) if dest_folder is not None else self.dest_folder
        destination.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar_ref:
            tar_ref.extractall(destination)

    def extract_archive(self, archive_path, dest_folder=None, delete_archive=False):
        archive_path = Path(archive_path)
        destination = Path(dest_folder) if dest_folder is not None else self.dest_folder

        if zipfile.is_zipfile(archive_path):
            self.unzip_file(archive_path, destination)
        elif tarfile.is_tarfile(archive_path):
            self.untar_file(archive_path, destination)
        else:
            raise ValueError(f"Unsupported archive format: {archive_path}")

        if delete_archive:
            self.delete_file(archive_path)

        return destination

    def move_files_and_cleanup(self, folder):
        folder = Path(folder)
        refs_equiv_path = folder / "refs_equiv"
        refs_sub_path = folder / "refs_subs"

        if refs_sub_path.exists():
            shutil.rmtree(refs_sub_path)

        if refs_equiv_path.exists():
            for item_path in refs_equiv_path.iterdir():
                shutil.move(str(item_path), str(folder))
            refs_equiv_path.rmdir()

    def process_unzipped_folders(self):
        for item_path in self.dest_folder.iterdir():
            if item_path.is_dir():
                self.move_files_and_cleanup(item_path)

    def unzip_all_in_folder(self):
        for item_path in self.dest_folder.iterdir():
            if zipfile.is_zipfile(item_path):
                self.extract_archive(item_path, delete_archive=True)
                self.process_unzipped_folders()

    def download_dataset(self, url):
        print("Downloading dataset...")
        zip_path = self.download_file(url)
        print("Uncompressing dataset...")
        self.extract_archive(zip_path, delete_archive=True)
        print("Processing unzipped folders...")
        self.unzip_all_in_folder()

    @staticmethod
    def has_subdirs(directory):
        return any(item.is_dir() for item in Path(directory).iterdir())

    @staticmethod
    def delete_file(file_path):
        Path(file_path).unlink()
