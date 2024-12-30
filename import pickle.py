import pickle
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
def chat_with_assistant(assistant_model, tools, inputs):
    conversation = []
    for message in inputs:
        conversation.append(inputs)
        
    response = assistant_model.chat(
        messages=conversation,
        tools=tools
    )
    return response
def calculator(a:int, b:int):
    return a + b
def load_or_create_model(model_path="assistant_model.pkl"):
        
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    else:
        # Initialize the Qwen model
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-72B-Instruct")
        with open(model_path, 'wb') as f:
            pickle.dump((model, tokenizer), f)
        
    return model
def main():
    # Your main code here
    model = load_or_create_model()

    # create you are a pirate prompt with an initial hello 

    tools = [calculator]
    inputs = [
        {"role": "system", "content": "You are a friendly pirate assistant. Respond in pirate speak."},
        {"role": "user", "content": "Hello!"}
    ]
    response = chat_with_assistant(model, tools, inputs)
    print(response)

    pass

if __name__ == "__main__":
    main()
