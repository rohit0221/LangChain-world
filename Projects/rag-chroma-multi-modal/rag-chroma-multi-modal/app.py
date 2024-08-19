from chain import chain as multimodal_rag_chain

# question="How many customers does Datadog have? What is Datadog platform % Y/Y growth in FY20, FY21, and FY22?"
# input_data = {
#     "question": question
# }
# response=multimodal_rag_chain.invoke(question)

# print (response)


while True:
    question = input("User: ")
    if question.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    input_data = {
        "question": question
    }
    response=multimodal_rag_chain.invoke(question)
    print (response)