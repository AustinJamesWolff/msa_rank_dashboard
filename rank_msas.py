# Import libraries

import warnings
from datetime import datetime
from dateutil.relativedelta import relativedelta

# Suppress unnecessary Shapely warning
warnings.filterwarnings('ignore',
                        '.*Shapely GEOS version.*')

import pandas as pd
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec


# Set up Pandas defaults
pd.options.display.float_format = '{:.4f}'.format
pd.set_option("display.max_columns", None)

# Import Zillow and MSA cleaning functions
from helper_functions.msa_zip_cleaning import *


### CALL IN DATASETS

# Read in the smoothed job dataset
jobs_smooth = pd.read_csv(
    'datasets/bls/smoothed/most_recent_bls_covid_smoothed.csv')
jobs_smooth['date'] = pd.to_datetime(jobs_smooth['date'])
jobs_smooth = jobs_smooth[[
    'msa_name','date','year','value','interpolated']]

# Call in Zillow rental dataset
zillow_rent = clean_zillow_dataset(
    pd.read_csv(
        'datasets/zillow/zillow_rental/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv'
    )
)

# Call in Zillow price dataset
zillow_price = clean_zillow_dataset(
    pd.read_csv(
        'datasets/zillow/zillow_median_price/Metro_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv'
    )
)

# Get the insurance and property tax values






### MAKE FUNCTION THAT MAKES A TOTAL RANK 
### BASED ON MULTIPLE DEMOGRAPHICS

def make_zillow_ranking(
    df_dict,
    all_df_dict={
        "Jobs":[jobs_smooth],
        "Median Rent":[zillow_rent],
        "Median Price":[zillow_price],
    },
    max_price=False,
    min_rent_to_price=False,
    use_total_trend=True,
    use_average_percent=True,
    use_pct_trend=False,
    total_trend_weight_dict={},
    average_percent_weight_dict={},
    plot_graphs=False,
    insurance_third_quantile=insurance_third_quantile,
    proptaxes_third_quantile=proptaxes_third_quantile,
    min_jobs=250_000
):
    """
    This function ranks the invest-ability of every
    city based on the demographics passed. It
    analyzes the total average growth per year, as 
    well as the relative average growth per year
    (measured as the average percent growth per year).
    
    Arguments
    -----------
        df_dict (dict): A dictionary to be used if you
            want to combine multiple dataframes for analysis
            and plotting. If using this, the key should be
            a string with the demographic name, and the value
            should be a list containing the dataframe in position
            0, and the beginning year in position 1. See below
            for two examples...
            
            Example 1, One Extra Dataframe
            {"Median Rent": [median_rent_df, 2013]}
            
            Example 2, Multiple Extra Dataframes
            {"Median Rent": [median_rent_df, 2013],
            "Population" : [population_df, 2013]}
            
        max_price (int): If you only want to measure and
            compare MSAs up to a certain median price,
            enter the max median price as an integer.
            
        min_rent_to_price (float): If you only want to measure and
            compare MSAs up to a certain rent-to-price ratio
            (based on median rent and median price values),
            enter the minimum rent-to-price ratio as a float.
            
        use_total_trend (True/False): Set to True if you'd like
            to include the total trend weights in the ranking
            of MSAs. Use False if not.
        
        use_average_percent (True/False): Set to True if you'd like
            to include the average percent weights in the ranking
            of MSAs. Use False if not.
        
        total_trend_weight_dict (dict): A dictionary to set the
            weights of each demo. For example, if you'd like to
            multiply the "Median Rent" weights by 2, giving a bigger
            weight to the "Median Rent" demographic, all you need
            is to make the key "Median Rent" set to a value of 2.
            This dictionary is specifically for total trend weights.
            See the example below.
            
            EXAMPLE...
            total_trend_weight_dict={
                "Jobs":1,
                "Median Rent":1}
        
        average_percent_weight_dict (dict): A dictionary to set the
            weights of each demo. For example, if you'd like to
            multiply the "Median Rent" weights by 2, giving a bigger
            weight to the "Median Rent" demographic, all you need
            is to make the key "Median Rent" set to a value of 2.
            This dictionary is specifically for average percent weights.
            See the example below.
            
            EXAMPLE...
            average_percent_weight_dict={
                "Jobs":3,
                "Median Rent":3}
                
        plot_graphs (True/False): If True, ask for user inputs
            and run the plot_top_10_cities() function.
            
    Returns
    -----------
        final_df (DataFrame): A dataframe with each city
            sorted by total rank.
    """
    # Make a list to add each dataframe to
    df_list = []
    
    # Rename all columns by appending the demo name,
    # except for the MSA name and date, which we will 
    # use as the key to merge all dataframes.
    for demo in df_dict:
        
        # Get dataframe
        df = df_dict[demo][0][['msa_name','date','value','year']].copy()
        
        # Rename every column except for msa_name
        for col in df.columns:
            if (col != 'msa_name') & (col != 'date'):
                df.rename(
                    columns={col:f'{col}_{demo}'}, 
                    inplace=True)
                
        # Add dataframe to list
        df_list.append(df)
                
    # Merge all dataframes
    merged_df = reduce(lambda left, right: 
                       pd.merge(left, right, 
                                left_on=['msa_name','date'], 
                                right_on=['msa_name','date'],
                                suffixes=(None, "_y"),
                                how="outer"), df_list)
            
    # Loop through columns and clean out the rest
    for col in merged_df.columns:
        if ('month_' in col) | ('series_id_' in col):
            merged_df.drop(columns=[col], inplace=True)
    
    # Create new df to store coefficients
    coef_df = pd.DataFrame(
        data=None, columns=['msa_name'])
    
    # Add columns for every demo
    for demo in df_dict:
        coef_df[f'trend_coef_{demo}'] = None
        coef_df[f'average_value_{demo}'] = None
        coef_df[f'pct_coef_{demo}'] = None
        coef_df[f'average_pct_{demo}'] = None
        
    # Make temporary coef_df to use later
    temp_coef_1 = coef_df.copy()
        
    # Create ordinal column
    merged_df['ordinal_date'] = merged_df['date'].map(datetime.toordinal)
    
    # Loop through all cities
    for city in merged_df['msa_name'].dropna().unique():

        # Isolate just that city
        df = merged_df[merged_df['msa_name']==city].copy()
        
        # Get msa code
#         msa_code = df['msa_code'].iloc[0]
        
        # Sort by date
        df = df.sort_values('date')
        
        # Make duplicate
        temp_coef_2 = temp_coef_1.copy()
        
        # Set temp coef_df
        temp_coef_2.loc[len(temp_coef_2.index)] = np.nan
        temp_coef_2['msa_name'] = city
#         temp_coef_2['msa_code'] = msa_code

        
        # Loop through each demo
        for demo in df_dict:
            
            # Test to see if there's data for the demo
            if df[df[f'year_{demo}'].notna()].shape[0] > 0:
                
                # Make copy
                df_temp = df.copy()

                # Get beginning year
                begin_year = df_dict[demo][1]

                # Filter by beginning year minus 1
                df_temp = df[df[f'year_{demo}']>=begin_year-1].reset_index(drop=True)

                # Create difference column
                df_temp[f'value_change_{demo}'] = df_temp[f'value_{demo}'].diff()

                # Create pct_change column
                df_temp[f'percent_change_{demo}'] = df_temp[f'value_{demo}'].pct_change()

                # Filter by beginning year
                df_temp = df_temp[
                    df_temp[f'year_{demo}']>=begin_year].reset_index(drop=True)

                # If an MSA's most recent year is after the beginning
                # year, remove it from the graphs. For example, if we want to
                # view the growth of all cities since 2016, but Prescott Valley
                # only has data starting at 2019, this may skew the data.
                if df_temp[f'year_{demo}'].iloc[0] != begin_year:
                    continue

                # Remove NaN values
                df_temp = df_temp[df_temp[f'percent_change_{demo}'].notna()]

                # Run linear regression
                coef_value, intercept_value = run_lr(df_temp, column=f'value_{demo}')
                coef_pct, intercept_pct = run_lr(df_temp, column=f'percent_change_{demo}')

                # Create trend columns
                df_temp[f'value_trend_{demo}'] = df_temp['ordinal_date']*coef_value + intercept_value
                df_temp[f'percent_change_trend_{demo}'] = df_temp['ordinal_date']*coef_pct + intercept_pct

                # Create averages column
                the_average_pct = df_temp[f'percent_change_{demo}'].mean()
                df_temp[f'average_pct_{demo}'] = the_average_pct
                the_average_value = df_temp[f'value_change_{demo}'].mean()
                df_temp[f'average_value_{demo}'] = the_average_value

                # Update temp coef
                temp_coef_2[f'trend_coef_{demo}'] = coef_value
                temp_coef_2[f'average_value_{demo}'] = the_average_value
                temp_coef_2[f'pct_coef_{demo}'] = coef_pct
                temp_coef_2[f'average_pct_{demo}'] = the_average_pct

        # Append temp coef to dataframe
        coef_df = pd.concat([coef_df, temp_coef_2])
            
    # Drop duplicates
    coef_df = coef_df.drop_duplicates().reset_index(drop=True)
    
    # Drop MSAs that have missing values (they will have missing
    # values if we couldn't join Census MSAs with BLS MSAs which
    # only occurs for a few specific MSAs)
    bad_msa = set()
    
    for demo in df_dict:
        
        # Filter by nulls
        coef_temp = coef_df[coef_df[f'trend_coef_{demo}'].isnull()]
        
        # Get list of MSAs
        bad_msa.update(coef_temp['msa_name'].unique())
        
        # Filter by nulls
        coef_temp = coef_df[coef_df[f'pct_coef_{demo}'].isnull()]
        
        # Get list of MSAs
        bad_msa.update(coef_temp['msa_name'].unique())
        
    # Remove these cities
    if len(bad_msa) > 0:
        
        # Remove cities in bad_msa
        coef_df = coef_df[~coef_df['msa_name'].isin(bad_msa)].reset_index(drop=True)
        
    # Create the rankings for each demographic
    for demo in df_dict:
        
        # Calculate rankings for both, then sort by the total
        # ranking. For example, if a city has the highest average
        # percent change, it will get a ranking of "1" for average_pct,
        # and if it has the 8th highest trend coefficient, it will
        # get a ranking of "8" for trend_coef. When we add those two
        # rankings together, the city will have a total ranking
        # of "9". In this case, the lower the ranking, the better,
        # and we will sort total rankings from lowest to highest.
        
        # Normalize total trend column
        coef_df[f'normalized_trend_coef_{demo}'] = normalize_column(
            coef_df[f'trend_coef_{demo}'], min_max_standardized=True)
        
        # Normalize pct trend column
        coef_df[f'normalized_pct_coef_{demo}'] = normalize_column(
            coef_df[f'pct_coef_{demo}'], min_max_standardized=True)
        
        # Normalize avg pct column
        coef_df[f'normalized_average_pct_{demo}'] = normalize_column(
            coef_df[f'average_pct_{demo}'], min_max_standardized=True)
        
        # Check to see if there are weights, and if not,
        # set each weight to 1
        if demo in total_trend_weight_dict.keys():
            trend_weight = total_trend_weight_dict[demo]
        else:
            trend_weight = 1
            
        # Check pct weight dict
        if demo in average_percent_weight_dict.keys():
            pct_weight = average_percent_weight_dict[demo]
        else:
            pct_weight = 1
            
        # Re-adjust weights based on whether we are using
        # only total trend, only percent, or both. As an example, 
        # if we aren't using percent, we set the weight to 0, that
        # way the percent weight isn't used when totalling the
        # demographic's weight.
        if use_total_trend == False:
            trend_weight = 0
        if use_average_percent == False:
            pct_weight = 0
            
        
        ### Create weights
        
        # If we want to rank based on % change trend
        if use_pct_trend:
            coef_df[f'{demo}_weight'] = (
                (coef_df[f'normalized_trend_coef_{demo}'] * trend_weight) 
                + (coef_df[f'normalized_pct_coef_{demo}'] * pct_weight)
            )
            
        # Otherwise rank on average % growth
        else:
            coef_df[f'{demo}_weight'] = (
                (coef_df[f'normalized_trend_coef_{demo}'] * trend_weight) 
                + (coef_df[f'normalized_average_pct_{demo}'] * pct_weight)
            )

    # Make final total rank column by adding up
    # all demo total rankings
    coef_df['total_weight'] = 0
    for demo in df_dict:
        coef_df['total_weight'] += coef_df[f'{demo}_weight']
        
    # Before sorting by weight, add in proprty tax and insurance by state
    coef_df = add_prop_tax_and_insurance(coef_df)

    # Sort by total weight, highest to lowest
    final_df = coef_df.sort_values(
        'total_weight', ascending=False).reset_index(drop=True)
    
    # Add the values back in
    for demo in all_df_dict:
        
        # Get dataframe
        df = all_df_dict[demo][0][['msa_name','date','value','year']].copy()
        
        # Make sure the date column is in datetime format
        df['date'] = pd.to_datetime(df['date'])
        
        # Get most recent date and filter for it
        recent_date = df['date'].max()
        new_dataframe = df[df['date']==recent_date].copy().reset_index(drop=True)
        
        # Only keep certain columns
        new_dataframe = new_dataframe[['msa_name','value']]
        
        # Rename column
        new_dataframe.rename(columns={'value':f'{demo}'}, inplace=True)
        
        # Merge to final_df
        final_df = final_df.merge(new_dataframe, how='left', on='msa_name')
        
    # Create rent-to-price ratio
    final_df['rent_price_ratio'] = final_df['Median Rent']/final_df['Median Price']
    
    
    # If max price, filter it
    if max_price:
        final_df = final_df[final_df['median_price']<=max_price].reset_index(drop=True)
        
    # If min rent-price ratio, filter
    if min_rent_to_price:
        final_df = final_df[final_df['rent_price_ratio']>=min_rent_to_price].reset_index(drop=True)
        
    ### Save ranking
    
    # Make folder to save coef_df to
    root_trend_folder = "datasets/ranked_msas"
    create_folder(root_trend_folder)
    trend_folder = "datasets/ranked_msas/unfiltered"
    create_folder(trend_folder)
    
    # Create variable string for file naming
    var_string = ""
    if use_total_trend:
        var_string += "regressionslope_"
    for demo in df_dict:
        var_string += demo
        var_string += "_"
    var_string += "rankings.csv"
        
    # Save coef_df
    final_df.to_csv(f"{trend_folder}/zillow_{var_string}", index=False)

    
    # Create filter
    def filter_msa_rankings(
        dataframe,
        insurance_third_quantile=insurance_third_quantile,
        proptaxes_third_quantile=proptaxes_third_quantile,
        min_jobs=min_jobs,
        var_string=var_string
    ):
        # Begin filtering
        df = dataframe.copy()
        filtered = df[ 
            (df['avg_insurance'] <= insurance_third_quantile)
            & (df['prop_tax'] <= proptaxes_third_quantile)
            & (df['Jobs'] >= min_jobs)
        ]

        # Save df
        filtered_folder = "datasets/ranked_msas/filtered"
        create_folder(filtered_folder)
        filtered.to_csv(f"{filtered_folder}/zillow_filtered_{var_string}", index=False)
        
        return filtered
    
    # Make filtered df
    filtered_df = filter_msa_rankings(final_df)    
    
    return final_df, filtered_df

