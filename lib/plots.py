import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def update_prune_ratios(df):
    df = df.rename(columns={'Unnamed: 0': 'prune_ratio'})
    prune_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    df['prune_ratio'] = prune_ratios
    return df

def calculate_averages(df):
    accuracy_columns = [col for col in df.columns if col.endswith('_acc')]
    stderr_columns = [col for col in df.columns if col.endswith('_acc_stderr')]
    df['average_accuracy'] = df[accuracy_columns].mean(axis=1)
    df['average_stderr'] = df[stderr_columns].mean(axis=1)
    return df[['prune_ratio', 'average_accuracy', 'average_stderr']]

def prepare_language_data(df, language_code):
    accuracy_columns = [col for col in df.columns if col.endswith(f'_acc') and language_code in col]
    stderr_columns = [col for col in df.columns if col.endswith(f'_acc_stderr') and language_code in col]
    df[f'{language_code}_average_accuracy'] = df[accuracy_columns].mean(axis=1)
    df[f'{language_code}_average_stderr'] = df[stderr_columns].mean(axis=1)
    return df[['prune_ratio', f'{language_code}_average_accuracy', f'{language_code}_average_stderr']]

def plot_errorbar(combined_df_no_duplicates):
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

def plot_lineplot_with_confidence(combined_df_no_duplicates):
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=combined_df_no_duplicates, x='prune_ratio', y='average_accuracy', hue='method', 
                 style='method', markers=True, dashes=False, err_style="band")

    for method in combined_df_no_duplicates['method'].unique():
        method_data = combined_df_no_duplicates[combined_df_no_duplicates['method'] == method]
        plt.fill_between(method_data['prune_ratio'], method_data['lower'], method_data['upper'], alpha=0.2)

    plt.title('Average Accuracy by Sparsity Level for Each Pruning Method')
    plt.xlabel('Sparsity Ratio')
    plt.ylabel('Average Accuracy')
    plt.legend(title='Pruning Method')
    plt.show()

def plot_multi_line_language(language_data, languages):
    fig, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=False)  # Set sharey to False
    specific_xticks = [0.1, 0.3, 0.5, 0.7, 0.9]
    global_min, global_max = float('inf'), float('-inf')

    # First pass to determine the global y-limits
    for language_code in languages:
        for method in language_data[language_code]:
            method_data = language_data[language_code][method]
            global_min = min(global_min, method_data[f'{language_code}_average_accuracy'].min())
            global_max = max(global_max, method_data[f'{language_code}_average_accuracy'].max())

    # Define a margin for the y-limits
    margin = (global_max - global_min) * 0.1
    global_min -= margin
    global_max += margin

    # Second pass to plot and set uniform y-limits
    for i, language_code in enumerate(languages):
        ax = axes[i]
        ax.set_xticks(specific_xticks)
        ax.set_ylim(global_min, global_max)

        for method in language_data[language_code]:
            method_data = language_data[language_code][method]
            if len(method_data) == 1 and method == 'Base':
                ax.hlines(y=method_data[f'{language_code}_average_accuracy'].values, xmin=0, xmax=1,
                          colors='#d62728', linestyles='dotted', label=method, linewidth=2)
            else:
                sns.lineplot(ax=ax, data=method_data, x='prune_ratio',
                             y=f'{language_code}_average_accuracy', label=method)

        ax.set_title(f'Performance for {language_code.upper()} tasks')
        ax.set_xlim([0.1, 0.9])
        ax.set_xlabel('Prune Ratio')
        ax.set_ylabel('Average Accuracy')
        ax.legend(title='Pruning Method')

    plt.suptitle('Pruning Method Performance Split by Language')
    plt.tight_layout()
    plt.show()

def plot_grouped_bar_chart(grouped_data):
    # Filter out the 'Base' method data
    base_data = grouped_data[grouped_data['method'] == 'Base']
    non_base_data = grouped_data[grouped_data['method'] != 'Base']

    # Unique prune ratios and methods (excluding 'Base')
    unique_prune_ratios = non_base_data['prune_ratio'].unique()
    unique_methods = non_base_data['method'].unique()

    positions = np.arange(len(unique_prune_ratios))
    bar_width = 0.25

    plt.figure(figsize=(15, 8))

    # Plot the 'Base' method data as a horizontal line
    for _, row in base_data.iterrows():
        plt.axhline(y=row['average_accuracy'], color='darkred', linestyle='dotted', label='Base' if 'Base' not in plt.gca().get_legend_handles_labels()[1] else "")

    # Plot other methods
    for i, method in enumerate(unique_methods):
        method_data = non_base_data[non_base_data['method'] == method]
        plt.bar(positions + i * bar_width, method_data['average_accuracy'], width=bar_width, label=method,
                yerr=method_data['average_stderr'], capsize=5)

    plt.xticks(positions + bar_width * (len(unique_methods) - 1) / 2, unique_prune_ratios)
    plt.xlabel('Prune Ratio')
    plt.ylabel('Average Accuracy')
    plt.title('Average Accuracies of Different Pruning Methods at Each Pruning Ratio')
    plt.legend(title='Pruning Method')
    plt.show()

def main():
    magnitude_df = update_prune_ratios(pd.read_csv('../results/magnitude_results.csv'))
    sparsegpt_df = update_prune_ratios(pd.read_csv('../results/sparsegpt_results.csv'))
    wanda_df = update_prune_ratios(pd.read_csv('../results/wanda_results.csv'))
    base_df = pd.read_csv('../results/noprune_results.csv')
    base_df["prune_ratio"] = 0

    avg_magnitude_df = calculate_averages(magnitude_df)
    avg_sparsegpt_df = calculate_averages(sparsegpt_df)
    avg_wanda_df = calculate_averages(wanda_df)
    avg_base_df = calculate_averages(base_df)

    combined_df = pd.concat([
        avg_base_df.assign(method='Base'),
        avg_magnitude_df.assign(method='Magnitude'),
        avg_sparsegpt_df.assign(method='SparseGPT'),
        avg_wanda_df.assign(method='Wanda')
    ])

    languages = ['en', 'es', 'zh']
    language_data = {}
    for language_code in languages:
        language_data[language_code] = {
            'Magnitude': prepare_language_data(magnitude_df, language_code),
            'SparseGPT': prepare_language_data(sparsegpt_df, language_code),
            'Wanda': prepare_language_data(wanda_df, language_code),
            'Base': prepare_language_data(base_df, language_code)
        }

    combined_df['lower'] = combined_df['average_accuracy'] - combined_df['average_stderr']
    combined_df['upper'] = combined_df['average_accuracy'] + combined_df['average_stderr']

    grouped_data = combined_df.groupby(['prune_ratio', 'method']).agg({
        'average_accuracy': 'mean',
        'average_stderr': 'mean'
    }).reset_index()

    plot_grouped_bar_chart(grouped_data)
    plot_errorbar(combined_df)
    plot_lineplot_with_confidence(combined_df)
    plot_multi_line_language(language_data, languages)

if __name__ == "__main__":
    main()
