import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the CSV files into pandas dataframes
magnitude_df = pd.read_csv('../results/magnitude_results.csv')
sparsegpt_df = pd.read_csv('../results/sparsegpt_results.csv')
wanda_df = pd.read_csv('../results/wanda_results.csv')

# Define the function to update prune ratios
def update_prune_ratios(df):
    df = df.rename(columns={'Unnamed: 0': 'prune_ratio'})
    prune_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    df['prune_ratio'] = prune_ratios
    return df

# Updating prune ratios for each dataframe
magnitude_df = update_prune_ratios(magnitude_df)
sparsegpt_df = update_prune_ratios(sparsegpt_df)
wanda_df = update_prune_ratios(wanda_df)

# Calculating the average accuracy and average standard error for each pruning method and pruning ratio
def calculate_averages(df):
    accuracy_columns = [col for col in df.columns if col.endswith('_acc')]
    stderr_columns = [col for col in df.columns if col.endswith('_acc_stderr')]
    df['average_accuracy'] = df[accuracy_columns].mean(axis=1)
    df['average_stderr'] = df[stderr_columns].mean(axis=1)
    return df[['prune_ratio', 'average_accuracy', 'average_stderr']]

avg_magnitude_df = calculate_averages(magnitude_df)
avg_sparsegpt_df = calculate_averages(sparsegpt_df)
avg_wanda_df = calculate_averages(wanda_df)

# Creating a combined dataframe for the plot
combined_df = pd.concat([
    avg_magnitude_df.assign(method='Magnitude'),
    avg_sparsegpt_df.assign(method='SparseGPT'),
    avg_wanda_df.assign(method='Wanda')
])

# Removing any potential duplicates
combined_df_no_duplicates = combined_df.drop_duplicates(subset=['prune_ratio', 'method'])

# Plotting the errorbar plot
plt.figure(figsize=(12, 8))
for method in combined_df_no_duplicates['method'].unique():
    method_data = combined_df_no_duplicates[combined_df_no_duplicates['method'] == method]
    plt.errorbar(method_data['prune_ratio'], method_data['average_accuracy'], 
                 yerr=method_data['average_stderr'], label=method, fmt='o', alpha=0.8, linestyle='', capsize=3)

plt.title('Average Accuracy Across Tasks for Different Pruning Methods and Ratios')
plt.xlabel('Prune Ratio')
plt.ylabel('Average Accuracy')
plt.legend(title='Pruning Method')
plt.show()

# First, we calculate the confidence intervals for each method.
combined_df_no_duplicates['lower'] = combined_df_no_duplicates['average_accuracy'] - combined_df_no_duplicates['average_stderr']
combined_df_no_duplicates['upper'] = combined_df_no_duplicates['average_accuracy'] + combined_df_no_duplicates['average_stderr']

# Now, we can plot using seaborn's lineplot with the confidence interval.
plt.figure(figsize=(14, 8))
sns.lineplot(data=combined_df_no_duplicates, x='prune_ratio', y='average_accuracy', hue='method', 
             style='method', markers=True, dashes=False, err_style="band")

# Adding the fill between the confidence intervals
for method in combined_df_no_duplicates['method'].unique():
    method_data = combined_df_no_duplicates[combined_df_no_duplicates['method'] == method]
    plt.fill_between(method_data['prune_ratio'], method_data['lower'], method_data['upper'], alpha=0.2)

plt.title('Average Accuracy by Sparsity Level for Each Pruning Method')
plt.xlabel('Sparsity Ratio')
plt.ylabel('Average Accuracy')
plt.legend(title='Pruning Method')
plt.show()

def prepare_language_data(df, language_code):
    accuracy_columns = [col for col in df.columns if col.endswith(f'_acc') and language_code in col]
    stderr_columns = [col for col in df.columns if col.endswith(f'_acc_stderr') and language_code in col]
    
    # Calculate average accuracy and standard error for the selected language
    df[f'{language_code}_average_accuracy'] = df[accuracy_columns].mean(axis=1)
    df[f'{language_code}_average_stderr'] = df[stderr_columns].mean(axis=1)
    return df[['prune_ratio', f'{language_code}_average_accuracy', f'{language_code}_average_stderr']]

# Prepare data for each language
languages = ['en', 'es', 'zh']  # English, Spanish, Chinese
language_data = {}
for language_code in languages:
    language_data[language_code] = {
        'Magnitude': prepare_language_data(magnitude_df, language_code),
        'SparseGPT': prepare_language_data(sparsegpt_df, language_code),
        'Wanda': prepare_language_data(wanda_df, language_code)
    }

# Plotting the multi-line plot split by language
fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)

for i, language_code in enumerate(languages):
    ax = axes[i]
    for method in language_data[language_code]:
        method_data = language_data[language_code][method]
        sns.lineplot(ax=ax, data=method_data, x='prune_ratio', 
                     y=f'{language_code}_average_accuracy', label=method)

    ax.set_title(f'Performance for {language_code.upper()} tasks')
    ax.set_xlabel('Prune Ratio')
    ax.set_ylabel('Average Accuracy' if i == 0 else '')
    ax.legend(title='Pruning Method')

plt.suptitle('Pruning Method Performance Split by Language')
plt.tight_layout()
plt.show()
