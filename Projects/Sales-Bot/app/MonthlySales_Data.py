from data_models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
model = ChatOpenAI(model="gpt-4o-mini",temperature=0)

# And a query intented to prompt a language model to populate the data structure.
MonthlySalesData_query = '''Please provide the monthly sales data for apple inc the years 2020, 2021, and 2022. The data should include the following fields for each month:\n
- `month`: The name or number of the month.
- `sales_volume_units`: The total number of units sold.
- `revenue_million_usd`: The total revenue in millions of USD.
- `average_price_per_unit_usd`: The average price per unit in USD.

The data should cover every month of the years 2020, 2021, and 2022. Please organize the data in a tabular format or list each month's data separately, including the year.
'''

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=MonthlySalesData)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

MonthlySalesData_chain = prompt | model | parser



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import os

@tool
def fetch_data_by_year(year):
    """
    Fetches data for a specific year from the DataFrame called "MonthlySales_df"
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    year (int): The year for which data is required.
    
    Returns:
    pd.DataFrame: Filtered DataFrame containing data for the specified year.
    """
    df = pd.read_csv('MonthlySales_Data.csv')
    return df[df['month'].str.contains(str(year))].reset_index(drop=True)

@tool
def fetch_data_by_month_year(month, year):
    """
    Fetches data for a specific month and year from the DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    month (str): The month for which data is required (e.g., 'January').
    year (int): The year for which data is required.
    
    Returns:
    pd.DataFrame: Filtered DataFrame containing data for the specified month and year.
    """
    df = pd.read_csv('MonthlySales_Data.csv')
    return df[(df['month'].str.contains(month)) & (df['month'].str.contains(str(year)))].reset_index(drop=True)

@tool
def fetch_data_by_date_range(start_date, end_date):
    """
    Fetches data for a specific date range from the DataFrame.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    start_date (str): The start date of the range (e.g., 'January 2020').
    end_date (str): The end date of the range (e.g., 'December 2022').
    
    Returns:
    pd.DataFrame: Filtered DataFrame containing data within the specified date range.
    """
    df = pd.read_csv('MonthlySales_Data.csv')
    return df[(df['month'] >= start_date) & (df['month'] <= end_date)].reset_index(drop=True)

@tool
def plot_sales_revenue():
    """
    Plots a line chart of monthly sales volume and revenue.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    
    Returns:
    matplotlib.figure.Figure: The generated figure object.
    """
    df = pd.read_csv('MonthlySales_Data.csv')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='month', y='sales_volume_units', marker='o', label='Sales Volume', ax=ax)
    sns.lineplot(data=df, x='month', y='revenue_million_usd', marker='o', label='Revenue', ax=ax)
    plt.xticks(rotation=45)
    plt.title('Monthly Sales Volume and Revenue')
    plt.xlabel('Month')
    plt.ylabel('Amount')
    plt.legend()
    return fig

@tool
def plot_monthly_revenue():
    """
    Plots a bar chart of monthly revenue.
    
    Returns:
    matplotlib.figure.Figure: The generated figure object.
    """
    df = pd.read_csv('MonthlySales_Data.csv')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=df, x='month', y='revenue_million_usd', palette='viridis', ax=ax)
    plt.xticks(rotation=45)
    plt.title('Monthly Revenue')
    plt.xlabel('Month')
    plt.ylabel('Revenue (Million USD)')
    return fig

@tool
def plot_avg_price_per_unit():
    """
    Plots a line chart of the average price per unit over time.
    
    Returns:
    matplotlib.figure.Figure: The generated figure object.
    """
    df = pd.read_csv('MonthlySales_Data.csv')
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(data=df, x='month', y='average_price_per_unit_usd', marker='o', color='orange', ax=ax)
    plt.xticks(rotation=45)
    plt.title('Average Price per Unit Over Time')
    plt.xlabel('Month')
    plt.ylabel('Average Price per Unit (USD)')
    return fig

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

@tool
def plot_heatmap_sales_revenue():
    """
    Creates a heatmap of monthly sales data and returns it as a base64 encoded string.

    Returns:
    dict: A dictionary containing the base64 encoded image and a message.
    """
    try:
        df = pd.read_csv('MonthlySales_Data.csv')
        df_pivot = df.pivot_table(values='revenue_million_usd', index='month', columns='average_price_per_unit_usd')
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(df_pivot, cmap='YlGnBu', annot=True, fmt='.1f', linewidths=.5, ax=ax)
        ax.set_title('Heatmap of Monthly Revenue by Average Price Per Unit')
        ax.set_xlabel('Average Price Per Unit (USD)')
        ax.set_ylabel('Month')

        # Save the plot to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        
        # Encode the bytes as base64
        img_str = base64.b64encode(buf.getvalue()).decode()
        
        plt.close(fig)  # Close the figure to prevent display in notebooks
        
        return {
            "image": img_str,
            "message": "Heatmap figure created successfully. Use the provided base64 string to display the image."
        }
    except Exception as e:
        return {
            "image": None,
            "message": f"Failed to create heatmap: {str(e)}"
        }

@tool
def plot_correlation_heatmap():
    """
    Plots a heatmap showing the correlation between sales volume, revenue, and average price per unit.
    
    Returns:
    matplotlib.figure.Figure: The generated figure object.
    """
    df = pd.read_csv('MonthlySales_Data.csv')
    corr = df[['sales_volume_units', 'revenue_million_usd', 'average_price_per_unit_usd']].corr()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt='.2f', linewidths=0.5, ax=ax)
    plt.title('Correlation Heatmap')
    return fig

@tool
def monthly_summary():
    """
    Provides a summary of monthly statistics including mean and sum of sales volume, revenue, and average price per unit.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    
    Returns:
    pd.DataFrame: A DataFrame containing the mean and sum statistics for each month.
    """
    df = pd.read_csv('MonthlySales_Data.csv')
    summary = df.groupby(df['month'].str.split(' ', 1).str[1]).agg({
        'sales_volume_units': ['mean', 'sum'],
        'revenue_million_usd': ['mean', 'sum'],
        'average_price_per_unit_usd': 'mean'
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()






@tool
def yearly_summary():
    """
    Provides a summary of yearly statistics including mean and sum of sales volume, revenue, and average price per unit.
    
    Args:
    df (pd.DataFrame): The DataFrame containing the sales data.
    
    Returns:
    pd.DataFrame: A DataFrame containing the mean and sum statistics for each year.
    """
    df = pd.read_csv('MonthlySales_Data.csv')
    summary = df.groupby(df['month'].str.split(' ', 1).str[1]).agg({
        'sales_volume_units': ['mean', 'sum'],
        'revenue_million_usd': ['mean', 'sum'],
        'average_price_per_unit_usd': 'mean'
    })
    summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
    return summary.reset_index()

# List of all functions with summaries
MonthlySales_tools = [
    fetch_data_by_year,
    fetch_data_by_month_year,
    fetch_data_by_date_range,
    plot_sales_revenue,
    plot_monthly_revenue,
    plot_avg_price_per_unit,
    plot_heatmap_sales_revenue,
    plot_correlation_heatmap,
    monthly_summary,
    yearly_summary
]

llm = ChatOpenAI(model="gpt-4o-mini")


llm_with_tools = llm.bind_tools(MonthlySales_tools)

from langchain import hub

# Get the prompt to use - you can modify this!
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages

from langchain.agents import create_tool_calling_agent

MonthlySales_agent = create_tool_calling_agent(llm, MonthlySales_tools, prompt)

from langchain.agents import AgentExecutor

MonthlySales_agent_executor = AgentExecutor(agent=MonthlySales_agent, tools=MonthlySales_tools, verbose=True)