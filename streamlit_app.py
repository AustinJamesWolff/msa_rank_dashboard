import streamlit as st
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

import pandas as pd
import plotly
import math
from functools import reduce

# Import everything from our ranking file
from rank_msas import *

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
st.sidebar.header('Real Estate MSA Dashboard')

st.sidebar.subheader('MSA Ranking Parameters')
datasets_to_use = st.sidebar.multiselect(
    label='Select datasets to rank by', 
    options=['Jobs', 'Price', 'Rent'], 
    default=['Jobs', 'Price', 'Rent'],
    )

# Get the earliest year from the jobs dataset
earliest_year = jobs_smooth['date'].min().year

# Get the most recent year
most_recent_year = jobs_smooth['date'].max().year

# Define the slider for user input of start year
start_year_to_use = st.sidebar.slider('Specify start year', earliest_year, most_recent_year, earliest_year)

# Create the df_dict to be used in the ranking function
df_dict = {}
for dataset_chosen in datasets_to_use:
    if dataset_chosen == 'Jobs':
        df_dict[dataset_chosen] = [jobs_smooth, start_year_to_use]
    elif dataset_chosen == 'Price':
        df_dict[dataset_chosen] = [zillow_price, start_year_to_use]
    elif dataset_chosen == 'Rent':
        df_dict[dataset_chosen] = [zillow_rent, start_year_to_use]

# Now create the sidebar elements that allow for weights
# to be assigned based on whether to use just average percent growth
# or average growth by year
st.sidebar.markdown('''
    The ranking algorithm needs to know if it should rank by the average monthly growth, or by the average *percent* growth. The average monthly growth will tend to favor MSAs with higher values, while the average percent growth will tend to favor MSAs that grew the fastest relative to their size (even if they are small). You can always filter the results by size after they are ranked.
    ''')
total_or_percent_growth = st.sidebar.selectbox(
    'Rank by average growth or average percent growth',
    ['Avg. Monthly Growth',
    'Avg. Percent Growth'])

# Create a slider to select weights
weight_dict={}
for demo in datasets_to_use:
    demo_weight = st.sidebar.slider(
        f'Specify how important {demo} is', 
        1, 5, 1)
    weight_dict[demo] = demo_weight

# Create the variables for the filter function
if total_or_percent_growth == 'Avg. Monthly Growth':
    use_total_trend = True
    use_average_percent = False
    total_trend_weight_dict = weight_dict
    average_percent_weight_dict = {}
elif total_or_percent_growth == 'Avg. Percent Growth':
    use_total_trend = False
    use_average_percent = True
    total_trend_weight_dict = {}
    average_percent_weight_dict = weight_dict


# Run the ranking algorithm based on the
# inputs specified in the sidebar
rank_all = make_zillow_ranking(
    df_dict=df_dict,
    all_df_dict={
        "Jobs":[jobs_smooth],
        "Rent":[zillow_rent],
        "Price":[zillow_price],
    },
    use_total_trend=use_total_trend,
    use_average_percent=use_average_percent,
    total_trend_weight_dict=total_trend_weight_dict,
    average_percent_weight_dict=average_percent_weight_dict
)

# Get a list of weights used for column names in the dataframe
weight_col_names = [f'{demo}_weight' for demo in datasets_to_use]

# Create filtering
st.sidebar.subheader('MSA Filtering Parameters')

# Filter by minimum job count
min_jobs = st.sidebar.number_input(
    label='Minimum number of jobs',
    min_value=0,
    step=100_000)

# Filter by maximum price
get_biggest_price_num = int(math.ceil(
    rank_all['Price'].max() / 1000000.0) * 1000000)
max_price = st.sidebar.number_input(
    label='Maximum price',
    max_value=get_biggest_price_num,
    min_value=0,
    value=get_biggest_price_num,
    step=int(100_000))

# Adjust rank dataframe proptax
rank_all['prop_tax'] = rank_all['prop_tax'] * 100

# Filter out high property tax
get_biggest_tax = float(
    math.ceil(rank_all['prop_tax'].max()))
max_tax = st.sidebar.number_input(
    label=f'Maximum property tax (%)',
    max_value=get_biggest_tax,
    min_value=0.0,
    value=get_biggest_tax,
    step=0.1)

# Filter out high insurance
get_biggest_insurance = int(math.ceil(
    insurance['avg_insurance'].max() / 1000.0) * 1000)
max_insurance = st.sidebar.number_input(
    label=f'Maximum insurance',
    max_value=get_biggest_insurance,
    min_value=0,
    value=get_biggest_insurance,
    step=int(500))

# Now filter the dataframe
rank_all = rank_all[
    (rank_all['Jobs'] >= min_jobs)
    & (rank_all['Price'] <= max_price)
    & (rank_all['avg_insurance'] <= max_insurance)
    & (rank_all['prop_tax'] <= max_tax)
]

st.sidebar.markdown('''
---
Created with ❤️ by [Austin James Wolff](https://www.linkedin.com/in/austin-james-wolff/).
''')


# Row A - Ranked MSAs DataFrame
with st.container():
    st.markdown('### Ranked MSAs')
    st.dataframe(data=rank_all[
            ['rank','msa_name'] + weight_col_names +
            ['total_weight','avg_insurance','prop_tax',
             'Jobs','Rent','Price','rent_price_ratio']],
             hide_index=True
        )

### Row B - Plot Top MSA Demographics

# Clean the jobs_smooth dataset to only include the top
# 5 MSAs in our rank_all dataframe
top_5_msas = list(rank_all['msa_name'].head(5))
job_df_list = []
for msa in top_5_msas:
    temp_df = jobs_smooth[
        jobs_smooth['msa_name']==msa].copy()
    temp_df.rename(columns={'value':f'{msa} Jobs'}, inplace=True)

    # Sort by date
    temp_df = temp_df.sort_values('date')

    # Drop unnecessary columns
    temp_df.drop(columns=['msa_name','year','interpolated'], inplace=True)
    
    job_df_list.append(temp_df)
msas_to_plot_jobs = reduce(
    lambda x, y: pd.merge(x, y, on='date'), job_df_list)

with st.container():
    st.markdown('#### Job Growth')
    st.line_chart(
            msas_to_plot_jobs,
            x='date'
            )

    # st.markdown('#### M-o-M Job Percent Change')

### COMMENTING OUT M-O-M GRAPHS
# # Create columns
# cols = st.columns(5)

# # Adjust dataframe for col m-o-m plotting:
# msas_to_plot_job_percent = msas_to_plot_jobs.copy()

# # Only keep data past 2021 to account for COVID anomaly
# msas_to_plot_job_percent = msas_to_plot_job_percent[
#     msas_to_plot_job_percent['date'] >= '2021-01-01']

# # Now loop through MSAs to plot M-o-M
# for i in range(len(top_5_msas)):
#     msa = top_5_msas[i]
#     # Add percent change column
#     msas_to_plot_job_percent[f'{msa} % Change'] = msas_to_plot_job_percent[f'{msa} Jobs'].pct_change()
#     msas_to_plot_job_percent.drop(columns=[f'{msa} Jobs'], inplace=True)

#     with cols[i]:
#         st.markdown(f'{msa}')
#         st.line_chart(
#                 msas_to_plot_job_percent,
#                 x='date',
#                 y=f'{msa} % Change',
#                 height=200
#                 )


### Now add Price Growth
# Get one dataframe with top cities for price
price_df_list = []
for msa in top_5_msas:
    temp_df = zillow_price[
        zillow_price['msa_name']==msa].copy()
    temp_df.rename(columns={'value':f'{msa} Price'}, inplace=True)

    # Sort by date
    temp_df = temp_df.sort_values('date')

    # Drop unnecessary columns
    temp_df.drop(columns=['msa_name','year'], inplace=True)
    
    price_df_list.append(temp_df)
msas_to_plot_price = reduce(
    lambda x, y: pd.merge(x, y, on='date'), price_df_list)

# Now plot Price data
with st.container():
    st.markdown('#### Price Appreciation')
    st.line_chart(
            msas_to_plot_price,
            x='date'
            )

### Now add Rent Growth
# Get one dataframe with top cities for price
rent_df_list = []
for msa in top_5_msas:
    temp_df = zillow_rent[
        zillow_rent['msa_name']==msa].copy()
    temp_df.rename(columns={'value':f'{msa} Rent'}, inplace=True)

    # Sort by date
    temp_df = temp_df.sort_values('date')

    # Drop unnecessary columns
    temp_df.drop(columns=['msa_name','year'], inplace=True)
    
    rent_df_list.append(temp_df)
msas_to_plot_rent = reduce(
    lambda x, y: pd.merge(x, y, on='date'), rent_df_list)

# Now plot Price data
with st.container():
    st.markdown('#### Rent Growth')
    st.line_chart(
            msas_to_plot_rent,
            x='date'
            )
   