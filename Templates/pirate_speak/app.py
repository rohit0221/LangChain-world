from  pirate_speak import chain

# Take input from the user
# user_input = input("Enter a sentence to translate into pirate speak: ")




while True:
    user_input = input("Enter a sentence to translate into pirate speak:: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    # Run the chain with the user's input
    response = chain.invoke({"text": user_input, "chat_history": []})
    print(response.content)