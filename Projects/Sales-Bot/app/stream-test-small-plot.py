python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data
data = {
    'month': ['January 2020', 'February 2020', 'March 2020', 'April 2020', 'May 2020'],
    'sales_volume_units': [50000, 45000, 60000, 55000, 70000],
    'revenue_million_usd': [1500, 1350, 1800, 1650, 2100],
    'average_price_per_unit_usd': [30, 30, 30, 30, 30]
}

df = pd.DataFrame(data)

# Create a pivot table for heatmap
heatmap_data = df.pivot_table(index='month', values='revenue_million_usd', columns='average_price_per_unit_usd', fill_value=0)

# Create a heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', cbar_kws={'label': 'Revenue (Million USD)'})
plt.title('Heatmap of Sales Revenue')
plt.xlabel('Average Price per Unit (USD)')
plt.ylabel('Month')
plt.show()