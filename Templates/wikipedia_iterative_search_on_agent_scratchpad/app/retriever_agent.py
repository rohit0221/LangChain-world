from langchain.agents import AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from agent_scratchpad import format_agent_scratchpad
from output_parser import parse_output
from prompts import retrieval_prompt
from wikipedia_retriever import retriever_description, search

prompt = ChatPromptTemplate.from_messages(
    [
        ("user", retrieval_prompt),
        ("ai", "{agent_scratchpad}"),
    ]
)
prompt = prompt.partial(retriever_description=retriever_description)

model = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, max_tokens=1000
)

chain = (
    RunnablePassthrough.assign(
        agent_scratchpad=lambda x: format_agent_scratchpad(x["intermediate_steps"])
    )
    | prompt
    | model.bind(stop=["</search_query>"])
    | StrOutputParser()
)

agent_chain = (
    RunnableParallel(
        {
            "partial_completion": chain,
            "intermediate_steps": lambda x: x["intermediate_steps"],
        }
    )
    | parse_output
)

executor = AgentExecutor(agent=agent_chain, tools=[search], verbose=True)