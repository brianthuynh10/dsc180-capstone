import pandas as pd 
import numpy as np 
import regex as re
import h5py as h5
import os

""""
 Cleans the BNPP reports data by extracting FINDINGS section and linking to HDF5 files
 Purpose: 
    The file create will be used to link the reports to the images in HDF5 files for later processing
        Such as feeding into LLM to generate labels
"""
def parse_reports(text: str) -> str:
    """
    Docstring for parse_reports
    
    :param text: Description
    :type text: str
    :return: Description
    :rtype: str
    """
    match = re.search(r'FINDINGS.*', text, re.DOTALL)
    return match.group(0) if match else None

def connect_data(phonetic_name, hdf5_names, base="~/teams/b1/"):
    """
    Docstring for connect_data
    
    :param phonetic_name: Description
    :param hdf5_names: Description
    :param base: Description
    """
    base = os.path.expanduser(base)

    for name in hdf5_names:
        path = os.path.join(base, f"{name}.hdf5")

        if not os.path.exists(path):
            continue

        with h5py.File(path, "r") as f:
            for key in f.keys():
                if phonetic_name.lower() in key.lower():
                    return name  # <-- this is the file it lives in

    return None

def main():
    # Load BNPP reports from Teams folder: 
    bnpp_reports = pd.read_csv(
        '~/teams/b1/bnpp-reports-clean.csv',
        usecols=['Phonetic', 'ReportClean']
    )
    bnpp_reports = bnpp_reports.sample(n=10_000, random_state=42)    # Create list of hdf5 names to search through
    # hdf5 file names
    hdf5_names = [f'bnpp_frontalonly_1024_{i}' for i in range(1,11)]
    hdf5_names.append('bnpp_frontalonly_1024_0_1')
    # parse the reports (save space)
    bnpp_reports['ReportClean'] = bnpp_reports['ReportClean'].apply(parse_reports)
    # drop the NA because they aren't able to work with LLM 
    bnpp_reports.dropna(subset=['ReportClean'], inplace=True)

    # this will take a minute
    bnpp_reports['hdf5_file_name'] = bnpp_reports['Phonetic'].apply(lambda x: connect_data(x, hdf5_names))
    # drop NA because they couldn't be found in any hdf5 file
    bnpp_reports.dropna(subset=['hdf5_file_name'], inplace=True)
    # save the cleaned data
    bnpp_reports.to_csv('bnpp_reports_with_hdf5.csv', index=False)
    
