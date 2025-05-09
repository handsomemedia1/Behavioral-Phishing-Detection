import seaborn as sns
import pandas as pd

data = pd.DataFrame({
    'Category': ['Labor Participation', 'Unemployment'],
    'USA': [67, 4.6],
    'Germany': [75.5, 3.2]
})

data_melt = data.melt(id_vars='Category', var_name='Country', value_name='Value')
sns.barplot(x='Category', y='Value', hue='Country', data=data_melt)
plt.title('Labor Market Integration (2024)')
plt.ylabel('Percentage (%)')
plt.show()