from agent import agent, tools
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Tuple
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool




class AgentInput(BaseModel):
    input: str
    chat_history: List[Tuple[str, str]] = Field(
        ..., extra={"widget": {"type": "chat", "input": "input", "output": "output"}}
    )


agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True).with_types(
    input_type=AgentInput
)


while True:
    user_input = input("Enter the shopping item:")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    response = agent_executor.invoke({"input": user_input})