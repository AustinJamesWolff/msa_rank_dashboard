import streamlit as st
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

import pandas as pd
import numpy as np
import math
from functools import reduce
import sys
import path
import os

# dir = path.Path(__file__).abspath()
# sys.path.append(dir.parent.parent)

# Import our ranked file
ranked_msas = pd.read_csv("bigger_score/outputs/biggerscore_v2.csv", dtype={'msa_code':str})

with open('bigger_score/style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('BiggerScore')

# st.sidebar.subheader('MSA Ranking Parameters')
# datasets_to_use = st.sidebar.multiselect(
#     label='Select datasets to rank by', 
#     options=['Jobs', 'Price', 'Rent'], 
#     default=['Jobs', 'Price', 'Rent'],
#     )

# # Get the earliest year from the jobs dataset
# earliest_year = jobs_smooth['date'].min().year

# # Get the most recent year
# most_recent_year = jobs_smooth['date'].max().year

# # Define the slider for user input of start year
# start_year_to_use = st.sidebar.slider('Specify start year', earliest_year, most_recent_year, earliest_year)

# # Create the df_dict to be used in the ranking function
# df_dict = {}
# for dataset_chosen in datasets_to_use:
#     if dataset_chosen == 'Jobs':
#         df_dict[dataset_chosen] = [jobs_smooth, start_year_to_use]
#     elif dataset_chosen == 'Price':
#         df_dict[dataset_chosen] = [zillow_price, start_year_to_use]
#     elif dataset_chosen == 'Rent':
#         df_dict[dataset_chosen] = [zillow_rent, start_year_to_use]



### Create a dictionary to hold the future weights of the RANK columns
# Rank Weights
rank_weights = {

        'RANK_5-Year Household Growth': 2,
        'RANK_5-Year Population Growth': 2,
        'RANK_5-Year Job Growth': 2,
        'RANK_5-Year Job Growth x Normalized': 5,
        'RANK_Income': 2,
        'RANK_5-Year Income Growth': 1,
        'RANK_Unemployment_Rate': 2,
        'RANK_Vacancy_Rate': 2,
        'Population_Size_Category': 250,
        'RANK_1-Year Price Forecast': 2,

        'RANK_Permits_as_Percent_of_Total_Units': 2,
        'RANK_1-Year_HH_Growth_Minus_Percent_New_Supply': 2,

        'RANK_housecanary_rentpriceratio': 3,
        'RANK_ACS_1_Year_Median_Price': 2,
        'RANK_insurance': 5,
        'RANK_Median_Prop_Tax_Rate': 5,
    
}



# Create a slider to select weights
weight_dict={}
for demo in rank_weights:
    if demo == 'Population_Size_Category':
        multiplier = 1
    else:
        multiplier = 1
    demo_weight = st.sidebar.slider(
        f'Specify how important {demo} is', 
        0 * multiplier, 10 * multiplier, value=rank_weights[demo] * multiplier)
    weight_dict[demo] = demo_weight


# Create a Landlord Friendliness Modifier
ranked_msas['Landlord_Friendliness_Modifier'] = np.where(
    ranked_msas['landlord_friendly_or_not'] == 'Landlord Friendly',
    0.1,
    np.where(
        ranked_msas['landlord_friendly_or_not'] == 'Landlord Semi-Friendly',
        0,
        -0.1
    )
)

### Create the basic sums of the ranks
max_score = 0
display_msa = ranked_msas.copy()
display_msa['Total_Rank_Sum'] = 0
for col in weight_dict:
    max_score += display_msa[col].max() * weight_dict[col]
    display_msa['Total_Rank_Sum'] += display_msa[col] * weight_dict[col]

# Create BiggerScore
display_msa['BiggerScore'] = display_msa['Total_Rank_Sum'] / max_score
display_msa['BiggerScore'] += (display_msa['Landlord_Friendliness_Modifier'] * display_msa['BiggerScore'])

### Now create ordinal categories -- perhaps group MSAs into Tiers
def create_ordinal_metrics(
        df,
        name_of_ordinal_column,
        column_to_measure,
        ascending=False
):
    

    if ascending:
        df[name_of_ordinal_column] = np.where(
            df[column_to_measure]>=df[column_to_measure].quantile(0.75),
            1,
            np.where(
                df[column_to_measure]>=df[column_to_measure].quantile(0.5),
                2,
                np.where(
                    df[column_to_measure]>=df[column_to_measure].quantile(0.25),
                    3,
                    4
                )
            )
        )
    else:
        df[name_of_ordinal_column] = np.where(
        df[column_to_measure]<=df[column_to_measure].quantile(0.25),
        1,
        np.where(
            df[column_to_measure]<=df[column_to_measure].quantile(0.5),
            2,
            np.where(
                df[column_to_measure]<=df[column_to_measure].quantile(0.75),
                3,
                4
            )
        )
    )

    return df

display_msa = create_ordinal_metrics(display_msa, "Job_Growth_Tier", "5-Year Job Growth")
display_msa = create_ordinal_metrics(display_msa, "Household_Growth_Tier", "5-Year Household Growth")
display_msa = create_ordinal_metrics(display_msa, "Population_Growth_Tier", "5-Year Population Growth")
display_msa = create_ordinal_metrics(display_msa, "Income_Growth_Tier", "5-Year Income Growth")
display_msa = create_ordinal_metrics(display_msa, "Vacancy_Rate_Tier", "Vacancy_Rate", ascending=True)
display_msa = create_ordinal_metrics(display_msa, "Price_Tier", "housecanary_median_price", ascending=True)


### Adjust BiggerScore v1 and v2 based on the tiers
def adjust_tier(
        df,
        tier_name
):
    
    df['BiggerScore'] = np.where(
        df[f'{tier_name}'] == 1,
        df['BiggerScore'] * 0.8,
        np.where(
            df[f'{tier_name}'] == 2,
            df['BiggerScore'] * 0.9,
            df['BiggerScore']
        )
    )

    return df

# display_msa = adjust_tier(display_msa, "Population_Growth_Tier")
display_msa = adjust_tier(display_msa, "Household_Growth_Tier")
display_msa = adjust_tier(display_msa, "Job_Growth_Tier")
display_msa = adjust_tier(display_msa, "Income_Growth_Tier")
display_msa = adjust_tier(display_msa, "Vacancy_Rate_Tier")
display_msa = adjust_tier(display_msa, "Price_Tier")

# Due to "punishing" metrics, the range of the scores decreases.
# We will arbitraily increase the range for presentability.
# Users may be more likely to trust market scores between 30%-75% than 10%-55%.
display_msa['BiggerScore'] += 0.15


# Create final rankings
display_msa = display_msa.sort_values("BiggerScore", ascending=False).reset_index(drop=True)
display_msa['Final_Rank'] = display_msa.index + 1

# Reorganize columns
display_msa.drop(columns=['msa_code','Total_Rank_Sum'], inplace=True)
display_msa = display_msa[['msa_name_original','BiggerScore'] +
                          [col for col in display_msa if 
                                (col != 'msa_name_original') &
                                (col != 'BiggerScore')]]

display_msa = display_msa.sort_values("BiggerScore", ascending=False)
display_msa.reset_index(drop=True, inplace=True)



st.sidebar.markdown('''
---
Created with ❤️ by [Austin Wolff](https://www.linkedin.com/in/austin-james-wolff/).
''')


# Row A - Ranked MSAs DataFrame
with st.container():
    st.markdown('### Ranked MSAs')
    st.dataframe(data=display_msa, hide_index=False)
    # st.dataframe(data=ranked_msas[
    #         ['rank','msa_name'] + weight_col_names +
    #         ['total_weight','avg_insurance','prop_tax',
    #          'Jobs','Rent','Price','rent_price_ratio']],
    #          hide_index=True
    #     )

