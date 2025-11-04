
import pandas as pd
from pathlib import Path
import os
import requests
import zipfile
import shutil
import tarfile
from tqdm import tqdm
import yaml

## read data tables

na_vals = pd.io.parsers.readers.STR_NA_VALUES.difference({"NULL", "null", "n/a"})

def read_table(file_path: Path) -> pd.DataFrame:
    """Read tsv file as pandas dataframe without treating "null" as empty string."""

    file_path = str(file_path)

    sep = "\t" if file_path.endswith(".tsv") else ","
    return pd.read_csv(file_path, sep=sep, na_values=na_vals, keep_default_na=False)


## save dict to csv

def save_dict_to_csv(data: dict, file_path: Path, sep: str = ",", columns: list = ["key", "value"]):
    """Save a dictionary to a csv file."""
    pd.DataFrame(data.items(), columns=columns).to_csv(str(file_path), index=False, sep=sep)

## read yaml

def read_yaml(file_path: Path):
    with open(str(file_path), "r") as file:
        return yaml.safe_load(file)

## get data from urls

class DataDownloader:
    def __init__(self, dest_folder):
        self.dest_folder = dest_folder

    def download_file(self, url):
        local_filename = os.path.join(self.dest_folder, url.split('/')[-1])
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024
        t = tqdm(total=total_size, unit='iB', unit_scale=True)
        with open(local_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                t.update(len(chunk))
                f.write(chunk)
        t.close()
        return local_filename

    def unzip_file(self, zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.dest_folder)

    def untar_file(self, tar_path):
        with tarfile.open(tar_path, 'r:gz') as tar_ref:
            tar_ref.extractall(self.dest_folder)

    def move_files_and_cleanup(self, folder):
        refs_equiv_path = os.path.join(folder, 'refs_equiv')
        refs_sub_path = os.path.join(folder, 'refs_subs')

        if os.path.exists(refs_sub_path):
            shutil.rmtree(refs_sub_path)

        if os.path.exists(refs_equiv_path):
            for item in os.listdir(refs_equiv_path):
                item_path = os.path.join(refs_equiv_path, item)
                shutil.move(item_path, folder)
            os.rmdir(refs_equiv_path)

    def process_unzipped_folders(self):
        for item in os.listdir(self.dest_folder):
            item_path = os.path.join(self.dest_folder, item)
            if os.path.isdir(item_path):
                self.move_files_and_cleanup(item_path)

    def unzip_all_in_folder(self):
        for item in os.listdir(self.dest_folder):
            item_path = os.path.join(self.dest_folder, item)
            if zipfile.is_zipfile(item_path):
                self.unzip_file(item_path)
                self.delete_file(item_path)
                self.process_unzipped_folders()

    def download_dataset(self, url):
        print("Downloading dataset...")
        zip_path = self.download_file(url)
        print("Uncompressing dataset...")
        self.unzip_file(zip_path)
        print("Processing unzipped folders...")
        self.delete_file(zip_path)
        self.unzip_all_in_folder()

    @staticmethod
    def has_subdirs(directory):
        for item in os.listdir(directory):
            if os.path.isdir(os.path.join(directory, item)):
                return True
        return False
    
    @staticmethod
    def delete_file(file_path):
        os.remove(file_path)
