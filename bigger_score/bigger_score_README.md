# Shortcomings

### US Census MSA Year Comparisons

Some metros added counties to their metropolitan areas in 2023, artificially boosting its demogrpahic numbers such as population and households. There is no consistency across the MSA ACS survey, and it is unreliable when using it to compare years. After speaking with the US Census team over Slack, they suggested I use the ACS 5-Year survey to be able to retrieve data at the county level, and sum those metrics for all counties within a given MSA, for all MSAs, using a county-to-MSA crosswalk file. Unfortunately, the ACS 5-Year is always released later than 1-Year (in Decemeber, so at the time of writing this documentiation in October, we only have access to 2022 data). The ACS 5-Year also trades accuracy for availability, as the ACS 1-Year survey does not include any areas below 65,000 population, which accounts for quite a few counties. We also cannot compare the most recent year to the previous year. It is heavily suggested that the ACS 5-Year survey should only compare 5-year periods against other 5-year periods, i.e. it is fine to compare the 2018-2022 (2022) survey results with the 2013-2017 (2017) survey results, but not the 2018-2022 (2022) results with the 2017-2021 (2021) results. There must be no overlap.

This is quite a tradeoff, but in my opinion, our MSA data, while being a year more outdated, is now reliable enough to compare years, whereas the ACS 1-Year survey turned out to be completely unreliable due to certian MSAs adding counties to their geography, showing unnaturally boosts in population and household numbers. 



# Manual Tasks Needed

### US Census MSA Crosswalk File
Every year the Census changes a few MSA's geo_id codes. They also have a text-based crosswalk file for download for the most recent year here: https://www.census.gov/geographies/reference-files/time-series/geo/relationship-files.html

Whoever is in charge of the BiggerScore product will need to download this file once per year, **every March**, into the "helper_datasets" folder in the MARKET_INTELLIGENCE_ANALYSES repo, and update the code in the `functions_to_pull_data/census_functions.py` file, around line 47, to reflect this new file.