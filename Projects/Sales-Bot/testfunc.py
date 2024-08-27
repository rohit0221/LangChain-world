### 1. **Functions for Deep Insights**

#### Summary Statistics


import pandas as pd

def summarize_sales_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provides a summary of the sales data, including total, mean, median, and standard deviation.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    
    Returns:
    pd.DataFrame: A DataFrame summarizing total, mean, median, and standard deviation.
    """
    summary = {
        'Total Sales Volume': df['sales_volume_units'].sum(),
        'Mean Sales Volume': df['sales_volume_units'].mean(),
        'Median Sales Volume': df['sales_volume_units'].median(),
        'Std Dev Sales Volume': df['sales_volume_units'].std(),
        'Total Revenue': df['revenue_million_usd'].sum(),
        'Mean Revenue': df['revenue_million_usd'].mean(),
        'Median Revenue': df['revenue_million_usd'].median(),
        'Std Dev Revenue': df['revenue_million_usd'].std(),
        'Total Average Price': df['average_price_per_unit_usd'].mean()  # Assuming uniform price across months
    }
    return pd.DataFrame([summary])


#### Yearly Sales Summary


def yearly_sales_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provides a summary of sales data aggregated by year.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    
    Returns:
    pd.DataFrame: A DataFrame with total sales volume and revenue for each year.
    """
    df['Year'] = df['month'].str.extract(r'(\d{4})').astype(int)
    yearly_summary = df.groupby('Year').agg(
        Total_Sales_Volume=('sales_volume_units', 'sum'),
        Total_Revenue=('revenue_million_usd', 'sum')
    ).reset_index()
    return yearly_summary


#### Monthly Sales Trend


def monthly_sales_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Provides a monthly sales trend analysis.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    
    Returns:
    pd.DataFrame: A DataFrame with sales volume and revenue trends.
    """
    df['Month'] = pd.to_datetime(df['month'], format='%B %Y')
    monthly_trend = df[['Month', 'sales_volume_units', 'revenue_million_usd']].set_index('Month')
    return monthly_trend


### 2. **Graphical Visualizations**

#### Monthly Sales Volume and Revenue Trends


import matplotlib.pyplot as plt
import seaborn as sns

def plot_monthly_trends(df: pd.DataFrame):
    """
    Plots the monthly trends of sales volume and revenue.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    """
    df['Month'] = pd.to_datetime(df['month'], format='%B %Y')
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)
    sns.lineplot(data=df, x='Month', y='sales_volume_units', marker='o')
    plt.title('Monthly Sales Volume Trend')
    plt.xlabel('Month')
    plt.ylabel('Sales Volume Units')
    
    plt.subplot(2, 1, 2)
    sns.lineplot(data=df, x='Month', y='revenue_million_usd', marker='o', color='orange')
    plt.title('Monthly Revenue Trend')
    plt.xlabel('Month')
    plt.ylabel('Revenue (Million USD)')
    
    plt.tight_layout()
    plt.show()


#### Yearly Sales Summary Bar Chart


def plot_yearly_sales_summary(df: pd.DataFrame):
    """
    Plots the yearly sales volume and revenue summary.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the yearly sales summary data.
    """
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar', x='Year', y=['Total_Sales_Volume', 'Total_Revenue'], figsize=(10, 6))
    plt.title('Yearly Sales Volume and Revenue')
    plt.xlabel('Year')
    plt.ylabel('Amount')
    plt.legend(['Total Sales Volume', 'Total Revenue'])
    plt.tight_layout()
    plt.show()


#### Heatmap of Monthly Sales Data


def plot_heatmap(df: pd.DataFrame):
    """
    Plots a heatmap of monthly sales data.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    """
    df_pivot = df.pivot_table(values='revenue_million_usd', index='month', columns='average_price_per_unit_usd')
    plt.figure(figsize=(12, 8))
    sns.heatmap(df_pivot, cmap='YlGnBu', annot=True, fmt='.1f', linewidths=.5)
    plt.title('Heatmap of Monthly Revenue by Average Price Per Unit')
    plt.xlabel('Average Price Per Unit (USD)')
    plt.ylabel('Month')
    plt.show()


### 3. **Function List**


tools = [
    summarize_sales_data,
    yearly_sales_summary,
    monthly_sales_trend,
    plot_monthly_trends,
    plot_yearly_sales_summary,
    plot_heatmap
]

