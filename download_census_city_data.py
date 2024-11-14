"""
This module downloads the most recent census data from the
US Census at the city level.

You can simply run "python download_census_city_data.py" in
the terminal to begin the download. Make sure your
virtual environment is activated.
"""
# Import libraries
import pandas as pd

# All helper functions are in this module:
from helper_functions.census_functions import *

# Set up Pandas defaults
pd.options.display.float_format = '{:.4f}'.format
pd.set_option("display.max_columns", None)

# Create directories for file saving
create_folder("datasets/cleaned_census_api_files/")
create_folder("datasets/cleaned_census_api_files/graphable/")
create_folder("datasets/cleaned_census_api_files/raw/")
create_folder("datasets/cleaned_census_api_files/standardized/")

# Define beginning year
begin_year = 2010

# Ask for end year
end_year = int(input("What is the last year to download data from? "))

# Run the API to download population
population_city = download_and_format_msa_census_data(
    census_code="B01003_001E",
    census_code_meaning="population_city",
    format_msa=False,
    format_city=True,
    end_year=end_year)

# Run the API to download wage
median_income_city = download_and_format_msa_census_data(
    census_code="B19013_001E",
    census_code_meaning="median_income_city",
    format_msa=False,
    format_city=True,
    end_year=end_year)

# Run the API to download median price
median_price_city = download_and_format_msa_census_data(
    census_code="B25077_001E",
    census_code_meaning="median_price_city",
    format_msa=False,
    format_city=True,
    end_year=end_year)

# Run the API to download median rent
median_rent_city = download_and_format_msa_census_data(
    census_code="B25058_001E",
    census_code_meaning="median_rent_city",
    format_msa=False,
    format_city=True,
    end_year=end_year)

# Run the API to download total units
total_units_city = download_and_format_msa_census_data(
    census_code="B25001_001E",
    census_code_meaning="total_units_city",
    format_msa=False,
    format_city=True,
    end_year=end_year)

# Run the API to download vacant units
vacant_units_city = download_and_format_msa_census_data(
    census_code="B25002_003E",
    census_code_meaning="vacant_units_city",
    format_msa=False,
    format_city=True,
    end_year=end_year)

### Create Rent-to-Price dataset

# Rename columns
for i in range(2010, end_year + 1):
    median_rent_city.rename(columns={f"{i}":f"{i}_rent"}, inplace=True)

# Rename columns
for i in range(2010, end_year + 1):
    median_price_city.rename(columns={f"{i}":f"{i}_price"}, inplace=True)

# Merge price data
rent_to_price = median_rent_city.merge(
    median_price_city, how='inner', 
    on=['name','geo_id'])

# Loop through columns and divide rent by price per year
for i in range(2010, end_year + 1):
    rent_to_price[f'{i}'] = rent_to_price[f"{i}_rent"]/rent_to_price[f"{i}_price"]
    
    # Drop rent and price columns
    rent_to_price.drop(columns=[f'{i}_rent',f'{i}_price'], inplace=True)

# Save dataset
rent_to_price.to_csv(
    "datasets/cleaned_census_api_files/city_data/rent_to_price_ratio_city.csv", 
    index=False)

### CREATE VACANCY RATE
# Rename columns
for i in range(2010, end_year + 1):
    vacant_units_city.rename(columns={f"{i}":f"{i}_vacant"}, inplace=True)
    total_units_city.rename(columns={f"{i}":f"{i}_units"}, inplace=True)

# Enforce dtype
vacant_units_city['geo_id'] = vacant_units_city['geo_id'].astype(str)

# Merge price data
vacancy_rate = vacant_units_city.merge(
    total_units_city, how='inner', 
    on=['name','geo_id'])

# Loop through columns and divide rent by price per year
for i in range(2010, end_year + 1):
    vacancy_rate[f'{i}'] = vacancy_rate[f"{i}_vacant"]/vacancy_rate[f"{i}_units"]
    
    # Drop rent and price columns
    vacancy_rate.drop(columns=[f'{i}_vacant',f'{i}_units'], inplace=True)

# Save dataset
vacancy_rate.to_csv(
    "datasets/cleaned_census_api_files/city_data/vacancy_rate_city.csv", 
    index=False)