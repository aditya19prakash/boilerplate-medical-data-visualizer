import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1 - Import data
df = pd.read_csv("medical_examination.csv")

# 2 - Add 'overweight' column
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# 3 - Normalize data
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)


# 4 - Draw Categorical Plot
def draw_cat_plot():
    # 5 - Create DataFrame for cat plot using `pd.melt`
    df_cat = pd.melt(df,
                     id_vars=['cardio'],
                     value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])

    # 6 - Group and reformat the data
    df_cat = df_cat.value_counts().reset_index(name='total')

    # 7 - Rename columns for clarity
    df_cat.columns = ['cardio', 'variable', 'value', 'total']

    # 8 - Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(data=df_cat,
                      x='variable',
                      y='total',
                      hue='value',
                      col='cardio',
                      kind='bar').fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10 - Draw Heat Map
def draw_heat_map():
    # 11 - Clean the data
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12 - Calculate the correlation matrix
    corr = df_heat.corr()

    # 13 - Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14 - Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # 15 - Draw the heatmap
    sns.heatmap(corr,
                mask=mask,
                annot=True,
                fmt='.1f',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={'shrink': 0.5})

    # 16
    fig.savefig('heatmap.png')
    return fig
