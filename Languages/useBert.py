import torch
from pytorch_transformers import *
import numpy as np
import json

from collections import namedtuple

ListData = namedtuple('ListData',['id','labels'])

classes = [line.strip().split(":")[0] for line in open('labels.txt')]
#with open("something-something-v2-labels.json") as jsonfile:
#  json_reader = json.load(jsonfile)
#  for elem in json_reader:
#    print(elem)
#    classes.append(elem)
#sorted(classes)
for i in range(len(classes)):
  print(i,classes[i])

MODELS = [(BertModel, BertTokenizer, 'bert-large-uncased')]

for model_class, tokenizer_class, pretrained_weights in MODELS:
  #Load pretrained model/tokenizer
  tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
  model = model_class.from_pretrained(pretrained_weights)

  # Encode text
  for i in range(len(classes)):
    input_ids1 = torch.tensor([tokenizer.encode(classes[i])])

    with torch.no_grad():
      last_hidden_states = model(input_ids1)[0][0][-1]
      print(last_hidden_states.size())
      vec = last_hidden_states.numpy()
      print(np.sum(vec),i,classes[i])
#      np.save(str(i)+".npy",vec)
