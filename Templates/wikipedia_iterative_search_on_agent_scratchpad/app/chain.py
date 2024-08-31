from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import ConfigurableField

from prompts import answer_prompt
from retriever_agent import executor

prompt = ChatPromptTemplate.from_template(answer_prompt)

model = ChatOpenAI(
    model="gpt-4o-mini", temperature=0, max_tokens=1000
)

chain = (
    {"query": lambda x: x["query"], "information": executor | (lambda x: x["output"])}
    | prompt
    | model
    | StrOutputParser()
)

# Add typing for the inputs to be used in the playground


class Inputs(BaseModel):
    query: str


chain = chain.with_types(input_type=Inputs)

anthropic_iterative_search = chain.configurable_alternatives(
    ConfigurableField(id="chain"),
    default_key="response",
    # This adds a new option, with name `openai` that is equal to `ChatOpenAI()`
    retrieve=executor,
)