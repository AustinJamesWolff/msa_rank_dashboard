import streamlit as st
st.set_page_config(layout='wide', initial_sidebar_state='expanded')

import pandas as pd
import plost
import math

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


# Row A
with st.container():
    st.markdown('### Ranked MSAs')
    st.dataframe(data=rank_all[
            ['rank','msa_name'] + weight_col_names +
            ['total_weight','avg_insurance','prop_tax',
             'Jobs','Rent','Price','rent_price_ratio']],
             hide_index=True
        )

# Row B
seattle_weather = pd.read_csv('https://raw.githubusercontent.com/tvst/plost/master/data/seattle-weather.csv', parse_dates=['date'])
stocks = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/stocks_toy.csv')

c1, c2 = st.columns((7,3))
with c1:
    st.markdown('### Heatmap')
    plost.time_hist(
    data=seattle_weather,
    date='date',
    x_unit='week',
    y_unit='day',
    color=time_hist_color,
    aggregate='median',
    legend=None,
    height=345,
    use_container_width=True)
with c2:
    st.markdown('### Donut chart')
    plost.donut_chart(
        data=stocks,
        theta=donut_theta,
        color='company',
        legend='bottom', 
        use_container_width=True)

# Row C
st.markdown('### Line chart')
st.line_chart(seattle_weather, x = 'date', y = plot_data, height = plot_height)