import pathlib
import os

def create_data_directory():
    ## create a directory to save files if not existing already
    dir_path = pathlib.Path(__file__).parents[1]
    dir_data_files = str(dir_path)+'\\data_files'

    if os.path.isdir(dir_data_files):
        None
    else:
        #print("Creating Directory to store data")
        os.mkdir(dir_data_files)
        os.mkdir(dir_data_files+"\\raw")
        os.mkdir(dir_data_files+"\\processed")