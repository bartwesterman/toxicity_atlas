# Objective: perform synergy assessment of drug combinations based on the BLISS and Chi-square ratios

# Import libraries
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Read data
faers_md_data = pd.read_csv('data/01_md_data_init.csv')
faers_sd_data = pd.read_csv('data/01_sd_data_init.csv')

bench_data = pd.read_csv('data/01_kompas_benchmark_data_drug_ids.csv')
bench_data['snomed_id'] = bench_data['snomed_id'].astype(int)
bench_data = bench_data[['combination', 'snomed_id', 'bench_freq']]
bench_data = bench_data.drop_duplicates()

drug_names = pd.read_csv('data/01_drugs_ids.csv')
drug_names = drug_names[['id', 'name']]
drug_names = drug_names.drop_duplicates()

snomed_ids = pd.read_csv('data/01_snomed_ids.csv')  # used for decoding the adverse reactions in snomed ids
snomed_ids = snomed_ids.drop_duplicates(subset=['snomed_reaction'])

# MULTI DRUG DATA
# Group per tox id and snomed id the number of cases for FAERS md data
md_cases = faers_md_data.groupby(['tox_drug_id_y', 'snomed_reaction']).agg({'case_id': lambda x: list(x)}).reset_index()

# Calculate the total number of cases per tox id and snomed id for FAERS md data
md_cases['total_cases_md'] = md_cases['case_id'].apply(lambda x: len(x))

# Calculate the total number of cases solely per tox id for FAERS md data
# We consider also the entries with a total_cases below 6 for this step
total_md_cases_comb = md_cases.groupby(['tox_drug_id_y']).agg({'total_cases_md': lambda x: sum(x)})

# Merge total_md_cases_comb with md_cases
md_cases = pd.merge(md_cases, total_md_cases_comb, left_on='tox_drug_id_y', right_index=True)

# Rename the columns
md_cases = md_cases.rename(columns={'total_cases_md_x': 'total_cases_md_toxid_snomed',
                                    'total_cases_md_y': 'total_cases_md_toxid'})

# Calculate the relative frequency for each tox id and snomed id for FAERS md data
md_cases['rel_freq_md'] = md_cases['total_cases_md_toxid_snomed'] / md_cases['total_cases_md_toxid']  # y_obs
md_cases['rel_freq_md'] = md_cases['rel_freq_md'].apply(lambda x: round(x, 5))

md_cases = md_cases.reset_index(drop=True)

# Split tox_drug_id_y to be able to add single drug data
md_cases[['tox_drug_id_1', 'tox_drug_id_2']] = md_cases['tox_drug_id_y'].str.split('&', expand=True)

# Convert to int64
md_cases['tox_drug_id_1'] = md_cases['tox_drug_id_1'].astype('int64')
md_cases['tox_drug_id_2'] = md_cases['tox_drug_id_2'].astype('int64')

# SINGLE DRUG DATA
# Group per tox id and snomed id the number of cases for FAERS sd data
sd_cases = faers_sd_data.groupby(['tox_drug_id_y', 'snomed_reaction']).agg({'case_id': lambda x: list(x)}).reset_index()

# Calculate the total number of cases per tox id and snomed id for FAERS sd data
sd_cases['total_cases_sd'] = sd_cases['case_id'].apply(lambda x: len(x))

# Calculate the total number of cases solely per tox id for FAERS sd data
# We consider also the entries with a total_cases below 6 for this step
total_sd_cases_comb = sd_cases.groupby(['tox_drug_id_y']).agg({'total_cases_sd': lambda x: sum(x)})

# Merge total_sd_cases_comb with sd_cases
sd_cases = pd.merge(sd_cases, total_sd_cases_comb, left_on='tox_drug_id_y', right_index=True)

# Rename the columns
sd_cases = sd_cases.rename(columns={'total_cases_sd_x': 'total_cases_sd_toxid_snomed',
                                    'total_cases_sd_y': 'total_cases_sd_toxid'})

# Calculate the relative frequency for each tox id and snomed id for FAERS sd data
sd_cases['rel_freq_sd'] = sd_cases['total_cases_sd_toxid_snomed'] / sd_cases['total_cases_sd_toxid']
sd_cases['rel_freq_sd'] = sd_cases['rel_freq_sd'].apply(lambda x: round(x, 5))

sd_cases = sd_cases.reset_index(drop=True)

# COMMON MULTI DRUG AND SINGLE DRUG DATA
# Find common drugs based on sd ids and snomed ids, this is done twice because of the 2 drugs in 2 columns
common_md_sd_1 = pd.merge(md_cases, sd_cases, left_on=['tox_drug_id_1', 'snomed_reaction'],
                          right_on=['tox_drug_id_y', 'snomed_reaction'], how='inner')  # because in common between md and sd


common_md_sd_2 = pd.merge(common_md_sd_1, sd_cases, left_on=['tox_drug_id_2', 'snomed_reaction'],
                          right_on=['tox_drug_id_y', 'snomed_reaction'], how='inner')  # because in common between md and sd

# Rename columns
common_md_sd_2 = common_md_sd_2.rename(columns={'rel_freq_sd_x': 'rel_freq_sd_1',
                                                            'rel_freq_sd_y': 'rel_freq_sd_2'})
# Drop the columns of no relevance because are duplicates
common_md_sd = common_md_sd_2.drop(['tox_drug_id_y', 'tox_drug_id_y_y'], axis=1)

# Add a column to calculate bliss ratio
common_md_sd['sum_freq'] = (common_md_sd['rel_freq_sd_1'] + common_md_sd['rel_freq_sd_2']).round(5)
common_md_sd['mult_freq'] = (common_md_sd['rel_freq_sd_1'] * common_md_sd['rel_freq_sd_2']).round(5)
common_md_sd['y_pred'] = (common_md_sd['sum_freq'] - common_md_sd['mult_freq']).round(5)  # bliss_independence
common_md_sd['bliss_ratio'] = common_md_sd['rel_freq_md'] / common_md_sd['y_pred']  # y_obs / y_pred
common_md_sd['bliss_ratio'] = common_md_sd['bliss_ratio'].round(5)

# Drop the columns of no relevance
common_md_sd = common_md_sd.drop(['sum_freq', 'mult_freq'], axis=1)

# Rename the columns
common_md_sd = common_md_sd.rename(columns={'tox_drug_id_y_x': 'combination_tox_id',
                                            'rel_freq_md': 'y_obs',
                                            'case_id_x': 'case_id_md',
                                            'case_id_y': 'case_id_1',
                                            'case_id': 'case_id_2',
                                            'total_cases_sd_toxid_snomed_x': 'total_cases_sd_toxid_snomed_1',
                                            'total_cases_sd_toxid_x': 'total_cases_sd_toxid_1',
                                            'total_cases_sd_toxid_snomed_y': 'total_cases_sd_toxid_snomed_2',
                                            'total_cases_sd_toxid_y': 'total_cases_sd_toxid_2'})


### ADD DRUG NAMES
# Add the names to common_md_sd based on tox_id_combinations
common_md_sd = pd.merge(common_md_sd, drug_names, left_on='tox_drug_id_1', right_on='id',
                        how='left')  # add because to supplement the data
# Drop the tox_id_combinations column
common_md_sd = common_md_sd.drop('id', axis=1)

# Add the names to common_md_sd based on tox_id_combinations
common_md_sd = pd.merge(common_md_sd, drug_names, left_on='tox_drug_id_2', right_on='id',
                        how='left')  # add because to supplement the data
# Drop the tox_id_combinations column
common_md_sd = common_md_sd.drop('id', axis=1)

# Add a column to match the names of the drugs as a combination
common_md_sd['combination_name'] = common_md_sd['name_x'] + ' & ' + common_md_sd['name_y']

# Rename name_x and name_y columns
common_md_sd = common_md_sd.rename(columns={'name_x': 'drug_name_1', 'name_y': 'drug_name_2'})

# Reorder the columns
common_md_sd = common_md_sd[['combination_tox_id', 'combination_name', 'drug_name_1', 'drug_name_2',
                             'snomed_reaction', 'case_id_md',
                             'total_cases_md_toxid_snomed',
                             'total_cases_md_toxid', 'y_obs', 'tox_drug_id_1',
                             'case_id_1', 'total_cases_sd_toxid_snomed_1', 'total_cases_sd_toxid_1',
                             'rel_freq_sd_1', 'tox_drug_id_2', 'case_id_2', 'total_cases_sd_toxid_snomed_2',
                             'total_cases_sd_toxid_2', 'rel_freq_sd_2', 'y_pred', 'bliss_ratio']]

### ADD SNOMED IDS
common_md_sd = common_md_sd.merge(snomed_ids, on='snomed_reaction', how='left')  # add because to supplement the data

# Reorder the columns
common_md_sd = common_md_sd[['combination_tox_id', 'combination_name', 'drug_name_1', 'drug_name_2', 'snomed_reaction',
                             'meddra_preferred_term_name', 'meddra_high_level_term', 'meddra_high_level_term_name',
                             'case_id_md', 'total_cases_md_toxid_snomed', 'total_cases_md_toxid', 'y_obs',
                             'tox_drug_id_1', 'case_id_1', 'total_cases_sd_toxid_snomed_1', 'total_cases_sd_toxid_1', 'rel_freq_sd_1',
                             'tox_drug_id_2', 'case_id_2', 'total_cases_sd_toxid_snomed_2', 'total_cases_sd_toxid_2', 'rel_freq_sd_2',
                             'y_pred', 'bliss_ratio']]

# Sum up 1 and 2
common_md_sd['total_cases_sd_toxid_snomed_1_2'] = common_md_sd['total_cases_sd_toxid_snomed_1'] + common_md_sd['total_cases_sd_toxid_snomed_2']
common_md_sd['total_cases_sd_toxid_1_2'] = common_md_sd['total_cases_sd_toxid_1'] + common_md_sd['total_cases_sd_toxid_2']

# Find the no events number for chi square
common_md_sd['md_no_event'] = common_md_sd['total_cases_md_toxid'] - common_md_sd['total_cases_md_toxid_snomed']
common_md_sd['sd_no_event'] = common_md_sd['total_cases_sd_toxid_1_2'] - common_md_sd['total_cases_sd_toxid_snomed_1_2']

# Add chi square ratio column
common_md_sd['chi2_obs'] = common_md_sd['total_cases_md_toxid_snomed']/common_md_sd['md_no_event']
common_md_sd['chi2_exp'] = common_md_sd['total_cases_sd_toxid_snomed_1_2']/common_md_sd['sd_no_event']
common_md_sd['chi2_ratio'] = (common_md_sd['chi2_obs']) / (common_md_sd['chi2_exp'])

# Add a column with the observed values for chi2_contingency
common_md_sd = common_md_sd.assign(observed=common_md_sd.apply(lambda row: [[row['total_cases_md_toxid_snomed'],
                                                                             row['md_no_event']],
                                                                            [row['total_cases_sd_toxid_snomed_1_2'],
                                                                             row['sd_no_event']]], axis=1))

# Add columns for chi2_contingency and calculate the values
common_md_sd.loc[:, 'chi2'], common_md_sd.loc[:, 'p'], common_md_sd.loc[:, 'dof'], common_md_sd.loc[:, 'expected'] = zip(*common_md_sd['observed'].apply(lambda x: chi2_contingency(x)))  # correction=False

print('---Initial data---')
print('Number of unique drug combinations:', common_md_sd['combination_name'].nunique())
print('Number of unique SNOMED IDs:', common_md_sd['snomed_reaction'].nunique())
print('Number of total records:', len(common_md_sd['combination_name']))

common_md_sd.to_csv('data/02_data_init_bliss_chi.csv')


# Select columns for the final dataframe for the publication
common_md_sd = common_md_sd[['combination_name', 'drug_name_1', 'drug_name_2', 'snomed_reaction',
                             'meddra_preferred_term_name', 'meddra_high_level_term', 'meddra_high_level_term_name',
                             'total_cases_md_toxid_snomed', 'total_cases_md_toxid', 'y_obs',
                             'total_cases_sd_toxid_snomed_1', 'total_cases_sd_toxid_1',
                             'total_cases_sd_toxid_snomed_2', 'total_cases_sd_toxid_2',
                             'y_pred', 'bliss_ratio', 'chi2_obs', 'chi2_exp', 'chi2_ratio', 'chi2', 'p']]

# Rename the columns
common_md_sd.columns = ['combination_name', 'drug_name_1', 'drug_name_2', 'snomed_reaction',
                        'meddra_preferred_term_name', 'meddra_high_level_term', 'meddra_high_level_term_name',
                        'total_cases_md_snomed_drugs', 'total_cases_md_drugs', 'relative_frequency_md (y_obs)',
                        'total_cases_sd_snomed_drug1', 'total_cases_sd_drug_1',
                        'total_cases_sd_snomed_drug2', 'total_cases_sd_drug_2',
                        'bliss_independence (y_pred)', 'bliss_ratio', 'chi2_obs', 'chi2_exp', 'chi2_ratio', 'chi2', 'p']


# Filter by three columns>= 6 as a new dataframe
common_md_sd_min6 = common_md_sd[common_md_sd['total_cases_md_snomed_drugs'] >= 6]
common_md_sd_min6 = common_md_sd_min6[common_md_sd_min6['total_cases_sd_snomed_drug1'] >= 6]
common_md_sd_min6 = common_md_sd_min6[common_md_sd_min6['total_cases_sd_snomed_drug2'] >= 6]


print('---Data with a minimum of six cases in MD and SD---')
print('Number of unique drug combinations:', common_md_sd_min6['combination_name'].nunique())
print('Number of unique SNOMED IDs:', common_md_sd_min6['snomed_reaction'].nunique())
print('Number of total records:', len(common_md_sd_min6['combination_name']))
print('Number of records with Bliss ratio > 1:', len(common_md_sd_min6[common_md_sd_min6['bliss_ratio'] > 1]))
print('Number of records with a chi2 ratio > 1 & p-value < 0.05:', len(common_md_sd_min6[(common_md_sd_min6['chi2_ratio'] > 1) & (common_md_sd_min6['p'] < 0.05)]))


common_md_sd_min6.to_csv('data/03_data_init_bliss_chi_min6.csv')


# Add bench_freq column from bench_data to common_md_sd dataframe for common combination names and snomed IDs
common_md_sd_bench = common_md_sd.merge(bench_data, left_on=['combination_name', 'snomed_reaction'], right_on=['combination', 'snomed_id'], how='inner')
# Keep only the columns from common_md_sd dataframe and bench_freq column
common_md_sd_bench = common_md_sd_bench[['combination_name', 'drug_name_1', 'drug_name_2', 'snomed_reaction',
                                         'meddra_preferred_term_name', 'meddra_high_level_term', 'meddra_high_level_term_name',
                                         'total_cases_md_snomed_drugs', 'total_cases_md_drugs', 'relative_frequency_md (y_obs)',
                                         'total_cases_sd_snomed_drug1', 'total_cases_sd_drug_1',
                                         'total_cases_sd_snomed_drug2', 'total_cases_sd_drug_2',
                                         'bliss_independence (y_pred)', 'bliss_ratio', 'chi2_obs', 'chi2_exp', 'chi2_ratio', 'chi2', 'p', 'bench_freq']]
# Rename bench_freq column
common_md_sd_bench.columns = ['combination_name', 'drug_name_1', 'drug_name_2', 'snomed_reaction',
                              'meddra_preferred_term_name', 'meddra_high_level_term', 'meddra_high_level_term_name',
                              'total_cases_md_snomed_drugs', 'total_cases_md_drugs', 'relative_frequency_md (y_obs)',
                              'total_cases_sd_snomed_drug1', 'total_cases_sd_drug_1',
                              'total_cases_sd_snomed_drug2', 'total_cases_sd_drug_2',
                              'bliss_independence (y_pred)', 'bliss_ratio',
                              'chi2_obs', 'chi2_exp', 'chi2_ratio', 'chi2', 'p', 'Frequency in Pharmacotherapeutic compass']

print('---Benchmark dataset---')
print('Number of unique drug combinations:', common_md_sd_bench['combination_name'].nunique())
print('Number of unique SNOMED IDs:', common_md_sd_bench['snomed_reaction'].nunique())
print('Number of total records:', len(common_md_sd_bench['combination_name']))
print('Number of records with Bliss ratio > 1:', len(common_md_sd_bench[common_md_sd_bench['bliss_ratio'] > 1]))
print('Number of records with a chi2 ratio > 1 & p-value < 0.05:', len(common_md_sd_bench[(common_md_sd_bench['chi2_ratio'] > 1) & (common_md_sd_bench['p'] < 0.05)]))


common_md_sd_bench.to_csv('data/04_data_init_bliss_chi_bench.csv')

### FINAL DATASET
# Add a column with Y or N for rows that md_cases['total_cases_md_toxid_snomed'] >= 6 AND common_md_sd['total_cases_sd_toxid_snomed_1'] >= 6 AND common_md_sd['total_cases_sd_toxid_snomed_2'] >= 6
common_md_sd['Minim six cases'] = np.where((common_md_sd['total_cases_md_snomed_drugs'] >= 6) & (common_md_sd['total_cases_sd_snomed_drug1'] >= 6) & (common_md_sd['total_cases_sd_snomed_drug2'] >= 6), 'Y', 'N')


# Add Frequency in Pharmacotherapeutic compass from common_md_sd_bench dataframe to common_md_sd dataframe for common combination names and snomed ID
common_md_sd = common_md_sd.merge(bench_data, left_on=['combination_name', 'snomed_reaction'], right_on=['combination', 'snomed_id'], how='left')
# Drop combination and snomed_id columns
common_md_sd = common_md_sd.drop(['combination', 'snomed_id'], axis=1)
# Rename bench_freq into In Pharmacotherapeutic compass
common_md_sd = common_md_sd.rename(columns={'bench_freq': 'In Pharmacotherapeutic compass'})

print('---')
print('Number of records in Minim six cases:', len(common_md_sd[common_md_sd['Minim six cases'] == 'Y']))
print('Number of records in Pharmacotherapeutic compass:', len(common_md_sd[common_md_sd['In Pharmacotherapeutic compass'].notnull()]))
print('---')

# Save the final dataframe for the publication
common_md_sd.to_csv('data/05_data_final.csv')

print('DONE')
