from writer import chain as writer_chain
from researcher import chain as search_chain



from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnablePassthrough

chain_notypes = (
    RunnablePassthrough().assign(research_summary=search_chain) | writer_chain
)


class InputType(BaseModel):
    question: str


chain = chain_notypes.with_types(input_type=InputType)

while True:
    user_input = input("Enter the research topic:")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Run the chain with the user's input
    response = chain.invoke({"question": user_input})
    print(response)