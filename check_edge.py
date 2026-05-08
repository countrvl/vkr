import json
import pandas as pd

# Load coverage
with open('data/nl2sql/synthetic_ecommerce/coverage_summary.json', 'r') as f:
    coverage = json.load(f)

print('edge_type_distribution:', coverage['edge_type_distribution'])

edge_type_df = pd.DataFrame([
    {'edge_type': key, 'count': value}
    for key, value in coverage['edge_type_distribution'].items()
]).sort_values('count', ascending=False)

print('edge_type_df:')
print(edge_type_df)
print('dtypes:', edge_type_df.dtypes)
print('has NaN?', edge_type_df.isna().any().any())

# Load edge cases
with open('data/nl2sql/synthetic_ecommerce/edge_cases_v1.json', 'r') as f:
    edge_cases = json.load(f)

edge_queries_df = pd.DataFrame(edge_cases['queries'])
print('\nedge_queries_df edge_type column:')
print(edge_queries_df[['difficulty', 'group', 'edge_type']])
print('edge_type unique:', edge_queries_df['edge_type'].unique())

# Load dataset
with open('data/nl2sql/synthetic_ecommerce/dataset_v1.json', 'r') as f:
    dataset = json.load(f)

core_queries_df = pd.DataFrame(dataset['queries'])
print('\ncore_queries_df edge_type column (should be missing):')
print('edge_type' in core_queries_df.columns)

# Query results
query_results_df = pd.DataFrame(coverage['query_results'])
print('\nquery_results_df edge_type unique:', query_results_df['edge_type'].unique())
print('NaN count:', query_results_df['edge_type'].isna().sum())
print('Total rows:', len(query_results_df))

# Write results to file
with open('edge_check_results.txt', 'w') as f:
    f.write(str(edge_type_df))
    f.write('\n\n')
    f.write('Edge queries edge_type: ' + str(edge_queries_df['edge_type'].tolist()))
