from critique_revise_chain import critique_revise_chain

while True:
    user_input = input("Enter your query: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    input_data = {
    "query": user_input,
    "max_revisions": 2
    }
    response = critique_revise_chain.invoke(input_data)
    print(response)