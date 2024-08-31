from langchain.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough

from web import chain as search_chain
from writer import chain as writer_chain

chain_notypes = (
    RunnablePassthrough().assign(research_summary=search_chain) | writer_chain
)


class InputType(BaseModel):
    question: str


chain = chain_notypes.with_types(input_type=InputType)


if __name__ == "__main__":
    print(
        # chain.invoke({"question": "who is typically older: point guards or centers?"})
        chain.invoke({"question": "How many employees are there?"})
    )
