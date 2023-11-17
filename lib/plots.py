import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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
    accuracy_columns = [col for col in df.columns if col.endswith(f'_acc') and language_code in col and not col.startswith('headqa')]
    stderr_columns = [col for col in df.columns if col.endswith(f'_acc_stderr') and language_code in col and not col.startswith('headqa')]
    df[f'{language_code}_average_accuracy'] = df[accuracy_columns].mean(axis=1)
    df[f'{language_code}_average_stderr'] = df[stderr_columns].mean(axis=1)
    return df[['prune_ratio', f'{language_code}_average_accuracy', f'{language_code}_average_stderr']]

def update_and_average(df, prune_ratios, languages):
    df = df.rename(columns={'Unnamed: 0': 'prune_ratio'})
    df['prune_ratio'] = prune_ratios
    for lang in languages:
        acc_cols = [col for col in df.columns if col.endswith(f'{lang}_acc')]
        df[f'{lang}_average_accuracy'] = df[acc_cols].mean(axis=1)
    return df

def calculate_relative_accuracies(df, base_df, languages):
    for lang in languages:
        base_accuracy = base_df[f'{lang}_average_accuracy'][0]  # baseline accuracy
        df[f'{lang}_relative_accuracy'] = df[f'{lang}_average_accuracy'].apply(lambda x: (x - base_accuracy) / base_accuracy)
    return df

def plot_errorbar(df):
    plt.figure(figsize=(12, 8))
    for method in df['method'].unique():
        method_data = df[df['method'] == method]
        plt.errorbar(method_data['prune_ratio'], method_data['average_accuracy'], 
                     yerr=method_data['average_stderr'], label=method, fmt='o', alpha=0.8, linestyle='', capsize=3)

    plt.title('Average Accuracy Across Tasks for Different Pruning Methods and Ratios')
    plt.xlabel('Prune Ratio')
    plt.ylabel('Average Accuracy')
    plt.legend(title='Pruning Method')
    plt.show()

def plot_lineplot_with_confidence(df):
    plt.figure(figsize=(14, 8))
    
    non_base_df = df[df['method'] != 'Base']
    sns.lineplot(data=non_base_df, x='prune_ratio', y='average_accuracy', hue='method', 
                 style='method', markers=True, dashes=False, err_style="band")

    for method in non_base_df['method'].unique():
        method_data = non_base_df[non_base_df['method'] == method]
        plt.fill_between(method_data['prune_ratio'], method_data['lower'], method_data['upper'], alpha=0.2)

    base_df = df[df['method'] == 'Base']
    if not base_df.empty:
        base_accuracy = base_df['average_accuracy'].iloc[0]
        base_stderr = base_df['average_stderr'].iloc[0]
        plt.hlines(y=base_accuracy, xmin=0.1, xmax=0.9, color='darkred', linestyle='--', linewidth=2)
        plt.fill_betweenx(y=[base_accuracy - base_stderr, base_accuracy + base_stderr], x1=0.1, x2=0.9, color='darkred', alpha=0.2)
   
    plt.xticks([0.1, 0.3, 0.5, 0.7, 0.9])
    plt.xlim([0.09, 0.91])
 
    handles, _ = plt.gca().get_legend_handles_labels()
    base_legend = mlines.Line2D([], [], color='darkred', linestyle='--', linewidth=2, label='Base Model')
    handles.append(base_legend)

    plt.legend(handles=handles, title='Pruning Method', loc='lower left') 
    plt.title('Average Accuracy by Sparsity Level for Each Pruning Method')
    plt.xlabel('Sparsity Ratio')
    plt.ylabel('Average Accuracy')
    plt.show()

def plot_multi_line_language(language_data, languages):
    _, axes = plt.subplots(1, len(languages), figsize=(21, 7), sharey=True)

    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12
    suptitle_fontsize = 18

    for i, language_code in enumerate(languages):
        ax = axes[i]

        for method in language_data[language_code]:
            method_data = language_data[language_code][method]
            if len(method_data) == 1 and method == 'Base':
                ax.hlines(y=method_data[f'{language_code}_average_accuracy'].values, xmin=0, xmax=1,
                          colors='#d62728', linestyles='dotted', label=method, linewidth=2)
            else:
                sns.lineplot(ax=ax, data=method_data, x='prune_ratio',
                             y=f'{language_code}_average_accuracy', label=method)

        ax.set_title(f'Performance for {language_code.upper()} tasks', fontsize=title_fontsize)
        ax.set_xlim([0.1, 0.9])
        ax.set_ylim([0.4, 0.65])
        ax.yaxis.set_tick_params(labelleft=True)
        ax.set_xlabel('Prune Ratio', fontsize=label_fontsize)
        ax.set_ylabel('Average Accuracy', fontsize=label_fontsize)
        ax.legend(title='Pruning Method', fontsize=legend_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)

    plt.suptitle('Pruning Method Performance Split by Language', fontsize=suptitle_fontsize)
    plt.tight_layout()
    plt.show()

def plot_grouped_bar_chart(grouped_data):
    base_data = grouped_data[grouped_data['method'] == 'Base']
    non_base_data = grouped_data[grouped_data['method'] != 'Base']

    unique_prune_ratios = non_base_data['prune_ratio'].unique()
    unique_methods = non_base_data['method'].unique()

    positions = np.arange(len(unique_prune_ratios))
    bar_width = 0.25

    plt.figure(figsize=(15, 8))

    for _, row in base_data.iterrows():
        plt.axhline(y=row['average_accuracy'], color='darkred', linestyle='dotted', label='Base' if 'Base' not in plt.gca().get_legend_handles_labels()[1] else "")

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

def plot_relative_accuracy_subplots(dfs, languages):
    _, axes = plt.subplots(1, 3, figsize=(21, 7), sharey=True)

    title_fontsize = 16
    label_fontsize = 14
    tick_fontsize = 12
    legend_fontsize = 12
    suptitle_fontsize = 18

    colors = {'en': '#1f77b4', 'es': '#ff7f0e', 'zh': '#2ca02c'}
    language_labels = {'en': 'English', 'es': 'Spanish', 'zh': 'Chinese'}

    for ax, (method, df) in zip(axes, dfs.items()):
        for lang in languages:
            sns.lineplot(ax=ax, x='prune_ratio', y=f'{lang}_relative_accuracy', data=df, label=f'{language_labels[lang]}', color=colors[lang])
        ax.set_title(f'{method} Pruning', fontsize=title_fontsize)
        ax.set_xlabel('Prune Ratio', fontsize=label_fontsize)
        ax.set_ylabel('Relative Accuracy', fontsize=label_fontsize)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=tick_fontsize)
        ax.legend(title="Language", fontsize=legend_fontsize)

    plt.suptitle('Relative Accuracy from Baseline by Language for Different Pruning Methods', fontsize=suptitle_fontsize)
    plt.tight_layout()
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

    prune_ratios = [0.1, 0.3, 0.5, 0.7, 0.9]
    sparsegpt_df = update_and_average(sparsegpt_df, prune_ratios, languages)
    wanda_df = update_and_average(wanda_df, prune_ratios, languages)
    magnitude_df = update_and_average(magnitude_df, prune_ratios, languages)
    base_df = update_and_average(base_df, [0], languages)

    sparsegpt_relative = calculate_relative_accuracies(sparsegpt_df, base_df, languages)
    wanda_relative = calculate_relative_accuracies(wanda_df, base_df, languages)
    magnitude_relative = calculate_relative_accuracies(magnitude_df, base_df, languages)

    dfs = {
        'SparseGPT': sparsegpt_relative,
        'Wanda': wanda_relative,
        'Magnitude': magnitude_relative
    }

    plot_relative_accuracy_subplots(dfs, languages)

if __name__ == "__main__":
    main()
