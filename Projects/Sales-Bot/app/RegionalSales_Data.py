from data_models import *
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)


# And a query intented to prompt a language model to populate the data structure.
RegionalSalesData_query = '''Please provide the monthly sales data for apple inc the years 2020, 2021, and 2022. The data should include the following fields for each month:\n
- `month`: The name or number of the month.
- `sales_volume_units`: The total number of units sold.
- `revenue_million_usd`: The total revenue in millions of USD.
- `average_price_per_unit_usd`: The average price per unit in USD.

The data should cover every month of the years 2020, 2021, and 2022. Please organize the data in a tabular format or list each month's data separately, including the year.
'''

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=RegionalSalesData)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

RegionalSalesData_chain = prompt | model | parser


import pandas as pd

def fetch_data_by_year(df, year):
    return df[df['month'].str.contains(str(year))].reset_index(drop=True)

def fetch_data_by_month_year(df, month, year):
    return df[(df['month'].str.contains(month)) & (df['month'].str.contains(str(year)))].reset_index(drop=True)

def fetch_data_by_date_range(df, start_date, end_date):
    return df[(df['month'] >= start_date) & (df['month'] <= end_date)].reset_index(drop=True)

import seaborn as sns
import matplotlib.pyplot as plt
from langchain_core.tools import tool
