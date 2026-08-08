from src.agent import run_agent


print("🤖 AI Agent Started")
print("Type 'exit' to quit.\n")


while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("Goodbye!")
        break

    answer = run_agent(question)

    print("\nAgent:")
    print(answer)
    print("-" * 50)