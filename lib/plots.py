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

def plot_grouped_bar_chart(grouped_data):
    positions = np.arange(len(grouped_data['prune_ratio'].unique()))
    bar_width = 0.25

    plt.figure(figsize=(15, 8))

    for i, method in enumerate(grouped_data['method'].unique()):
        method_data = grouped_data[grouped_data['method'] == method]
        plt.bar(positions + i * bar_width, method_data['average_accuracy'], width=bar_width, label=method,
                yerr=method_data['average_stderr'], capsize=5)

    plt.xticks(positions + bar_width, grouped_data['prune_ratio'].unique())
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

    # Prepare data for each language
    languages = ['en', 'es', 'zh']
    language_data = {}
    for language_code in languages:
        language_data[language_code] = {
            'Magnitude': prepare_language_data(magnitude_df, language_code),
            'SparseGPT': prepare_language_data(sparsegpt_df, language_code),
            'Wanda': prepare_language_data(wanda_df, language_code),
            'Base': prepare_language_data(base_df, language_code)
        }

    # First, we calculate the confidence intervals for each method.
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
