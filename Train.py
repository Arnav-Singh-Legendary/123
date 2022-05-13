
from NeuralNetwork import bagOfWords, Tokenize, stem
from torch.utils.data import DataLoader, Dataset
from Brain import NeuraNet
import torch.nn as nn
from numpy import array as ppe
import numpy as np
import torch
import json

with open("Intents.json") as f:
    intents = json.load(f)

IgnoreWords = [',', '?', '!', '/', '.']
AllWords = []
Tags = []
xy = []

for intent in intents["intents"]:
    tag = intent["tag"]
    Tags.append(tag)
    
    for pattern in intent["patterns"]:
        w = Tokenize(pattern)
        AllWords.extend(w)
        xy.append((w, tag))

AllWords = [stem(w) for w in AllWords if w not in IgnoreWords]

Tags = sorted(set(Tags))
AllWords = sorted(set(AllWords))

Xtrain = []
Ytrain = []

for (pattern_sentence, tag) in xy:
    Bag = bagOfWords(pattern_sentence, AllWords)
    Xtrain.append(Bag)

    Label = Tags.index(tag)
    Ytrain.append(Label)


Xtrain = np.array(Xtrain)
Ytrain = np.array(Ytrain)

numEpochs = 1000
batchSize = 8
learningRate = 0.001
inputSize = len(Xtrain[0])
hiddenSize = 8
outputSize = len(Tags)

print("Training the Model...")

class ChatDataSet(Dataset):
    def __init__(self):
        self.nSamples = len(Xtrain)
        self.XData = Xtrain
        self.YData = Ytrain

    def __getitem__(self, index) :
        return self.XData[index],  Xtrain[index]
    
    def __len__(self):
        return self.nSamples

dataset = ChatDataSet()

trainLoader = DataLoader(dataset = dataset, batch_size = batchSize, shuffle = True, num_workers = 0)

device = torch.device('cpu')

model = NeuraNet(inputSize, hiddenSize, outputSize).to(device=device)

Criterion = nn.CrossEntropyLoss()
Optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)

for epoch in range(numEpochs):
    for (words, labels) in trainLoader:
        words = words.to(device)
        labels = labels.to(dtype = torch.long).to(device)
        outputs = model(words)
        loss = Criterion(outputs, labels)
        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch [{epoch}]/numEpochs: {numEpochs}, Loss = {loss.item():.4f}")

print(f"Final Loss: {loss.item():.}")