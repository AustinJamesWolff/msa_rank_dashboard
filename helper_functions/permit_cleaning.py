"""
This module cleans the raw permit data downloaded from
the "Building Permit Survery" section of the
U.S. Census's website. The website offers XLS
files for download from 2019-Present, and TXT
files before 2019. This module cleans both and
merges them, saving the end files to:

datasets/buidling_permits/cleaned/

For best results, run this file in the root
directory of the repo with:

python helper_functions/permit_cleaning.py
"""
import pandas as pd
from functools import reduce
from datetime import datetime
import re

# Import everything from our ranking file
from msa_zip_cleaning import clean_BLS_msa_names


# Combine all XLS Years
def combine_xls_unit_permits():

    # Get current year
    current_year = datetime.now().year

    # Create list to join dataframes
    unit_df_list = []

    # Read in each xls file - Data will be 
    # avilable up to previous year
    for i in range(2019, current_year):
        temp_unit_df = pd.read_excel(
            f'datasets/building_permits/raw/msaannual_{i}99.xls',
            sheet_name=0,
            skiprows=5
        ).dropna().rename(columns={'Name': 'msa_name'})

        # Clean column names
        temp_unit_df = clean_BLS_msa_names(temp_unit_df).drop(columns=[
            'CSA', 'CBSA', 'state', 'city'
        ])
        temp_unit_df.columns = [
            str(col) + f"_{i}" if col != 'msa_name' else col 
            for col in temp_unit_df.columns]

        # Append to our list
        unit_df_list.append(temp_unit_df)

    # Now merge all dataframes in our list
    combined_permit_xls_df = reduce(
        lambda x, y: pd.merge(x, y, on='msa_name'), unit_df_list)

    # Save combined dataframe
    combined_permit_xls_df.to_csv(
        'datasets/building_permits/cleaned/permitted_units_from_xls.csv',
        index=False)

    return


# Create function to clean and merge permit unit txt files
def combine_txt_unit_permits():

    # Create list to join dataframes
    unit_df_list = []

    # Read in each txt file
    for i in range(2014, 2019):
        # Read in the txt file - this will require some cleaning
        temp_unit_df = pd.read_fwf(
            f'datasets/building_permits/raw/tb3u{i}.txt',
            skiprows=4,
            skipfooter=2
            )

        # Rename certain rows to become column headers
        temp_unit_df.iloc[4, 5] = '2 Units'
        temp_unit_df.iloc[4, 7] = '3 and 4 Units'
        temp_unit_df.iloc[4, 8] = '5 Units or more'
        temp_unit_df.iloc[4, 9] = 'Num of Structures With 5 Units or more'

        # Drop the NaN columns
        temp_unit_df = temp_unit_df.drop([
            # temp_unit_df.columns[0], 
            temp_unit_df.columns[4],
            temp_unit_df.columns[6]
            ], axis=1)

        # Drop first few NaN rows
        temp_unit_df = temp_unit_df.drop([0, 1, 2, 3], axis=0)

        # Make new column header
        temp_unit_df.columns = temp_unit_df.iloc[0].to_list()
        temp_unit_df = temp_unit_df[1:]
        temp_unit_df.reset_index(drop=True, inplace=True)

        # Now clean the dataframe
        temp_unit_df = temp_unit_df.rename(columns={'Name': 'msa_name'})

        # Backfill NaNs in value columns
        cols = temp_unit_df.columns[2:].to_list()
        temp_unit_df.loc[:, cols] = temp_unit_df.loc[:, cols].bfill()

        # # Convert first two columns to lists, then iterate through them
        csa_list = temp_unit_df.iloc[:, 0].to_list()
        name_list = temp_unit_df.iloc[:, 1].to_list()

        # Algorithm to clean metro names
        new_name_list = []
        for j in range(len(csa_list)):
            metro_name = name_list[j]
            if isinstance(metro_name, str):
                metro_name = metro_name.strip("*")
                if re.match('[A-Z]{2}', str(csa_list[j])):
                    state_name = csa_list[j]
                    metro_name = name_list[j-1].strip(",*")
                    new_metro_name = metro_name + ", " + state_name
                    new_name_list[j-1] = new_metro_name
                    new_name_list.append(name_list[j])
                else:
                    new_name_list.append(metro_name)
            else:
                new_name_list.append(metro_name)

        # Assign cleaned names back to column
        temp_unit_df.iloc[:, 1] = new_name_list

        # Drop first column
        temp_unit_df = temp_unit_df.drop([
            temp_unit_df.columns[0]], axis=1)

        # Remove NaNs
        temp_unit_df = temp_unit_df.dropna().reset_index(drop=True)

        # Clean column names
        temp_unit_df = clean_BLS_msa_names(temp_unit_df).drop(columns=[
            'state', 'city'])

        # Remove NaNs
        temp_unit_df = temp_unit_df.dropna().reset_index(drop=True)

        temp_unit_df.columns = [
            str(col) + f"_{i}" if col != 'msa_name' else col 
            for col in temp_unit_df.columns]

        # Append to our list
        unit_df_list.append(temp_unit_df)

    # Now merge all dataframes in our list
    combined_permit_txt_df = reduce(
        lambda x, y: pd.merge(x, y, on='msa_name'), unit_df_list)

    # Save combined dataframe
    combined_permit_txt_df.to_csv(
        'datasets/building_permits/cleaned/permitted_units_from_txt.csv',
        index=False)

    return unit_df_list


if __name__ == '__main__':

    combine_xls_unit_permits()

    combine_txt_unit_permits()
