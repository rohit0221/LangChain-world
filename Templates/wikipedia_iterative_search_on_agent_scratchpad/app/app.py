from chain import anthropic_iterative_search

# Take input from the user
# user_input = input("Enter a sentence to translate into pirate speak: ")




while True:
    user_input = input("Enter your query ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Run the chain with the user's input
    response = anthropic_iterative_search.invoke({"query":user_input})
    # response = chain.invoke({"text": user_input, "chat_history": []})
    print(response.content)