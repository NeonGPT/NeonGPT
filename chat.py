# Import all the libaries
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# sets the device to cuda when its available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# loads the json file
def load_intents(file_path):
    with open(file_path, 'r') as json_data:
        intents = json.load(json_data)
    return intents

# lets the user set a custom name for the bot
bot_name = input("Bot Name: ")

# option to load a custom personality
print("Choose an intents JSON file:")
print("1. default.json (default)")
print("2. other file")
choice = input("Enter the number of your choice: ")

# the code for handeling the custom data
FILE = "./datasets/default.pth" 

if choice == "2":
    print("Enter file path for custom data")
    file_path = input()

    print("Enter path to .pth file. if you dont it use the train.bat")
    FILE = input()

else:
    file_path = "./datasets/default.json"




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
