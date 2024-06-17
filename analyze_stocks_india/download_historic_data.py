from datetime import date
from analyze_stocks_india.nse_archives import bhavcopy_save
import pandas as pd
from analyze_stocks_india.holidays import holidays
import time
from random import randint
import datetime
import os
import pathlib
import analyze_stocks_india.supporting_functions as sf

dir_path = pathlib.Path(__file__).parents[1]
dir_data_files = str(dir_path) + '\\data_files'

def date_to_csv_name(date):
    '''

    Parameters
    ----------
    date : class (datetime.date)
        DESCRIPTION.

        date is class object of datetime.date
    Returns
    -------
    a string in format of csv file name

    eg: cm01Apr2016bhav.csv

    '''

    def month_to_string(argument):
        switcher = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        return switcher.get(argument)

    month_ = month_to_string(date.month)

    day_ = str(date.day)
    if (len(day_) == 1):
        day_ = '0' + day_

    year_ = str(date.year)

    return ("cm{}{}{}bhav.csv".format(day_, month_, year_))


def update_raw_files(start_date, end_date):

    date_range = pd.bdate_range(start=start_date, end = end_date, freq='C', holidays = (holidays(2022)+holidays(2023)+holidays(2024)))
    #bdate = business days (weekends excluded by default)
    # start and end dates in "MM-DD-YYYY" format
    # holidays() function in (year,month) format
    #freq = 'C' is for custom

    dates = [x.date() for x in date_range]

    for i in range(len(dates)):
        date = dates[i]

        if os.path.isfile(dir_data_files +'\\' + 'raw' +'\\' +date_to_csv_name(date_range[i])) == False: # true if file does not exist
            print('Downloading file for date : {}'.format(date))
            try:
                bhavcopy_save(date,dir_data_files +'\\' + 'raw' +'\\')
                time.sleep(randint(3,4)) #adding random delay of 3-4 seconds
            except(ConnectionError) as e:
                time.sleep(10) #stop program for 10 seconds and try again
                try:
                    bhavcopy_save(date,dir_data_files +'\\' + 'raw' +'\\' )
                    time.sleep(randint(3,4))
                except (ConnectionError) as e:
                    print("{}: File not found".format(date))

    print("All Raw Files are Updated")
    print("\n\n\n")


def extract_data(filename, symbol):
    '''


    Parameters
    ----------
    filename : string
        DESCRIPTION: string of filename, note that the folder where csv files are saved is defined in function itself
    symbol : string
        DESCRIPTION:  name of share to extract values. note only Equity share value will be extracted

    Returns
    -------
    tuple of 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES'
    '''
    folder = dir_data_files +'\\' + 'raw' +'\\'
    data = pd.read_csv(folder + filename)
    data_filtered = data[data['SYMBOL'] == symbol]
    data_filtered = data_filtered[data_filtered['SERIES'] == 'EQ']

    to_export = []
    to_export.append(data_filtered['OPEN'].values[0])
    to_export.append(data_filtered['HIGH'].values[0])
    to_export.append(data_filtered['LOW'].values[0])
    to_export.append(data_filtered['CLOSE'].values[0])
    to_export.append(data_filtered['LAST'].values[0])
    to_export.append(data_filtered['PREVCLOSE'].values[0])
    to_export.append(data_filtered['TOTTRDQTY'].values[0])
    to_export.append(data_filtered['TOTTRDVAL'].values[0])
    to_export.append(data_filtered['TOTALTRADES'].values[0])

    return (to_export)


def extract_data_BE(filename, symbol):
    '''


    Parameters
    ----------
    filename : string
        DESCRIPTION: string of filename, note that the folder where csv files are saved is defined in function itself
    symbol : string
        DESCRIPTION:  name of share to extract values. note only Equity share value will be extracted

    Returns
    -------
    tuple of 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'TOTALTRADES'
    '''
    folder = dir_data_files +'\\' + 'raw' +'\\'
    data = pd.read_csv(folder + filename)
    data_filtered = data[data['SYMBOL'] == symbol]
    data_filtered = data_filtered[data_filtered['SERIES'] == 'BE']

    to_export = []
    to_export.append(data_filtered['OPEN'].values[0])
    to_export.append(data_filtered['HIGH'].values[0])
    to_export.append(data_filtered['LOW'].values[0])
    to_export.append(data_filtered['CLOSE'].values[0])
    to_export.append(data_filtered['LAST'].values[0])
    to_export.append(data_filtered['PREVCLOSE'].values[0])
    to_export.append(data_filtered['TOTTRDQTY'].values[0])
    to_export.append(data_filtered['TOTTRDVAL'].values[0])
    to_export.append(data_filtered['TOTALTRADES'].values[0])

    return (to_export)



def extract_and_save_data(symbol, start_date, end_date):
    date_range = pd.bdate_range(start=start_date, end=end_date, freq='C',
                                holidays=(holidays(2016) + holidays(2017) + holidays(2018) + holidays(2019) + holidays(
                                    2020) + holidays(2021) + holidays(2022) + holidays(2023)+holidays(2024)))

    # saving files for each symbol
    for s in symbol:
        print("Extracting data for {}".format(s))
        dict_frame = pd.DataFrame()
        for d in date_range:

            try:
                # print(d)
                filename = date_to_csv_name(d)
                data = [date(d.year, d.month, d.day)]

                #check for EQ series if not take BE Series
                try:
                    data_t = (extract_data(filename, s))
                except:
                    data_t = (extract_data_BE(filename, s))

                for i in data_t:
                    data.append(i)

                # print(d,data)
                df1 = pd.DataFrame([data],
                                   columns=['Date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE', 'TOTTRDQTY',
                                            'TOTTRDVAL', 'TOTALTRADES'])
                dict_frame = pd.concat([dict_frame, df1], ignore_index=True)
            except:
                print("Could not process for date {}".format(d))

        filename = '{}.csv'.format(s)
        dict_frame.to_csv((dir_data_files +'\\' + 'processed' +'\\'  + filename), index=False)
        #print("\n")


def extract_and_save_data_onlyUpdateNotAvailable(symbol, start_date, end_date):
    date_range = pd.bdate_range(start=start_date, end=end_date, freq='C',
                                holidays=(holidays(2016) + holidays(2017) + holidays(2018) + holidays(2019) + holidays(
                                    2020) + holidays(2021) + holidays(2022) + holidays(2023)+holidays(2024)))

    # saving files for each symbol
    c = 0
    l = len(symbol)
    for s in symbol:
        c += 1
        print("Processing {}/{}".format(c,l))

        filename_ = dir_data_files +'\\' + 'processed' +'\\' + "{}.csv".format(s)

        #check if valid data is present in file
        check_file_content = False
        try:
            df1 = pd.read_csv(filename_)
            if (df1.shape[0]>=1) :
                check_file_content = True
        except:
            None
        else:
            None


        # checking if file exists
        if (os.path.isfile(filename_) and check_file_content):

            # checking what data is available
            available_data_df = pd.read_csv( dir_data_files +'\\' + 'processed' +'\\' + "{}.csv".format(s))
            max_available_date = max(available_data_df['Date'])
            print("For {} data is available till {}, extracting for remaining dates".format(s, max_available_date))

            # converting max_available_date to right format for modifying_start_date
            t1 = max_available_date.split("-")
            new_start_date = "{}/{}/{}".format(t1[1],t1[2],t1[0])

            date_range_new = pd.bdate_range(start=new_start_date, end=end_date, freq='C',
                                        holidays=(holidays(2016) + holidays(2017) + holidays(2018) + holidays(
                                            2019) + holidays(
                                            2020) + holidays(2021) + holidays(2022) + holidays(2023)+holidays(2024)))

            print("Extracting data for {}".format(s))
            dict_frame_new = pd.DataFrame()
            for d in date_range_new:

                try:
                    # print(d)
                    filename = date_to_csv_name(d)
                    data = [date(d.year, d.month, d.day)]

                    # check for EQ series if not take BE Series
                    try:
                        data_t = (extract_data(filename, s))
                    except:
                        data_t = (extract_data_BE(filename, s))


                    for i in data_t:
                        data.append(i)

                    # print(d,data)
                    df1 = pd.DataFrame([data],
                                       columns=['Date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE',
                                                'TOTTRDQTY',
                                                'TOTTRDVAL', 'TOTALTRADES'])
                    dict_frame_new = pd.concat([dict_frame_new, df1], ignore_index=True)
                except:
                    print("Could not process for date {}".format(d))

            filename = '{}.csv'.format(s)
            dict_frame = pd.concat([available_data_df, dict_frame_new], ignore_index=True)
            dict_frame.to_csv((dir_data_files +'\\' + 'processed' +'\\'  + filename), index=False)
            print("\n")



        else:
            print("Extracting data for {}".format(s))
            dict_frame = pd.DataFrame()
            for d in date_range:

                try:
                    # print(d)
                    filename = date_to_csv_name(d)
                    data = [date(d.year, d.month, d.day)]
                    
                    # check for EQ series if not take BE Series
                    try:
                        data_t = (extract_data(filename, s))
                    except:
                        data_t = (extract_data_BE(filename, s))


                    for i in data_t:
                        data.append(i)

                    # print(d,data)
                    df1 = pd.DataFrame([data],
                                       columns=['Date', 'OPEN', 'HIGH', 'LOW', 'CLOSE', 'LAST', 'PREVCLOSE',
                                                'TOTTRDQTY',
                                                'TOTTRDVAL', 'TOTALTRADES'])
                    dict_frame = pd.concat([dict_frame, df1], ignore_index=True)
                except:
                    print("Could not process for date {}".format(d))

            filename = '{}.csv'.format(s)
            dict_frame.to_csv((dir_data_files +'\\' + 'processed' +'\\'  + filename), index=False)
            print("\n")




############################################################# Main Code ########################################

def download_raw_files(start_date='01/01/2023',N=10):
    '''
    Downloads the raw data files.
    start_date = 'mm/dd/yyyy'
    N --> specifies number of tries if connection got disconnected by the nse website
    Note: It's possible that nse website will disonnect the session and downloading will stop, run it couple of times to make sure all the files are downloaded
    Note: Folder structure for downloaded files is
            data_fiels
                raw
                processed
    '''
    # get today's date
    today = datetime.date.today()
    today_str = today.strftime("%m/%d/%y")

    #if today's time is less than 8pm, then change today's date to yesterday
    if (datetime.datetime.now().hour <=20 ):
        today = today - datetime.timedelta(days=1)
        today_str = today.strftime("%m/%d/%y")

    ############################################## Parameters Update Here #####################################################
    #start_date='01/01/2023' #start date for data extraction    M-d-y
    end_date=today_str #end date for data extraction    M-d-y

    for _ in range(N):
        try:
            update_raw_files(start_date, end_date)  # start and end dates in "MM-DD-YYYY" format
        except:
            print("Some error happend, rerun the downloading")
            print("Stopping downloading for 2 minutes")
            time.sleep(120)


def extract_new_data(symbol_list, start_date):
    '''
    Extracts the data from raw files and create a processed data file
    symbol_list --> must be a python list of nse symbols
    start_date = 'mm/dd/yyyy'
    Note: Folder structure for downloaded files is
            data_fiels
                raw
                processed
    '''
    # get today's date
    today = datetime.date.today()
    today_str = today.strftime("%m/%d/%y")

    # if today's time is less than 8pm, then change today's date to yesterday
    if (datetime.datetime.now().hour <= 20):
        today = today - datetime.timedelta(days=1)
        today_str = today.strftime("%m/%d/%y")

    end_date=today_str

    extract_and_save_data_onlyUpdateNotAvailable(symbol_list, start_date, end_date)



    #extract_and_save_data(symbol1, start_date, end_date)
    #extract_and_save_data(symbol2, start_date, end_date)
    #extract_and_save_data(symbol3, start_date, end_date)


def get_directory():
    print(dir_data_files)
    print(os.path.isdir(dir_data_files))
    print(dir_data_files +'\\' + 'raw')