from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
import pandas as pd

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)


from data_models import *
from MonthlySales_Data import *
from RegionalSales_Data import *

MonthlySales_df = pd.read_csv('MonthlySales_Data.csv')
RegionalSales_df = pd.read_csv('RegionalSales_Data.csv')

from MonthlySales_Data import *
# MonthlySales_agent_executor.invoke({
#     "input": "Give me the sales data for the year 2021."
# })


while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    MonthlySales_agent_executor.invoke({
    "input": user_input
})