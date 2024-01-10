import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Function to load intents from a JSON file
def load_intents(file_path):
    with open(file_path, 'r') as json_data:
        intents = json.load(json_data)
    return intents

print("Choose an intents JSON file:")
print("1. default.json (default)")
print("2. other file")

choice = input("Enter the number of your choice: ")

if choice == "2":
    print("Enter file path for custom data")
    file_path = input()
    
    print("Enter path to .pth file. If you don't, it will use the default.pth")
    pth_file_path = input()

    # Use the provided .pth file path, or default to "default.pth" if not provided
    FILE = pth_file_path.strip() if pth_file_path.strip() else "default.pth"

else:
    file_path = "default.json"
    FILE = "default.pth"

intents = load_intents(file_path)

data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Bot"
print("Let's chat! (type 'quit' to exit)")

while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
