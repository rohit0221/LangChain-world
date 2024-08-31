from csv_agent import agent_executor

while True:
    user_input = input("Enter your query: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    input_data = {
    "input": user_input,
    }
    response = agent_executor.invoke(input_data)
    print(response)