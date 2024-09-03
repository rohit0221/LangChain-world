from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI
import pandas as pd

model = ChatOpenAI(model="gpt-4o-mini",temperature=0)


from data_models import *
from MonthlySales_Data import *
from RegionalSales_Data import *
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents import AgentType

openai_api_key = os.environ.get('OPENAI_API_KEY')

MonthlySales_df = pd.read_csv('MonthlySales_Data.csv')
RegionalSales_df = pd.read_csv('RegionalSales_Data.csv')

print(MonthlySales_df)
from MonthlySales_Data import *
# MonthlySales_agent_executor.invoke({
#     "input": "Give me the sales data for the year 2021."
# })

llm = ChatOpenAI(
    temperature=0, model="gpt-4o-mini", openai_api_key=openai_api_key, streaming=True
)

pandas_df_agent = create_pandas_dataframe_agent(
    llm,
    MonthlySales_df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
    handle_parsing_errors=True,
)

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    pandas_df_agent.invoke({
    "input": user_input
})