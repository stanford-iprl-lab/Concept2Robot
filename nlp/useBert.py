import torch
from pytorch_transformers import *
import numpy as np

MODELS = [(BertModel, BertTokenizer, 'bert-large-uncased')]

for model_class, tokenizer_class, pretrained_weights in MODELS:
  #Load pretrained model/tokenizer
  tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
  model = model_class.from_pretrained(pretrained_weights)

  # Encode text
  input_ids1 = torch.tensor([tokenizer.encode("Move sth down")])

  with torch.no_grad():
    last_hidden_states = model(input_ids1)[0][0][-1]
    print(last_hidden_states.size())
    vec = last_hidden_states.numpy()
    print(np.sum(vec))
    #np.save("MoveSthUp.npy",vec)

  input_ids2 = torch.tensor([tokenizer.encode("Move sth up")])

  with torch.no_grad():
    last_hidden_states = model(input_ids2)[0][0][-1]
    print(last_hidden_states.size())
    vec = last_hidden_states.numpy()
    print(np.sum(vec))
    #np.save("MoveSthDown.npy",vec)

  input_ids3 = torch.tensor([tokenizer.encode("Pull sth from left to right")])

  with torch.no_grad():
    last_hidden_states3 = model(input_ids3)[0][0][-1]
    print(last_hidden_states3.size())
    vec3 = last_hidden_states3.numpy()
    print(np.sum(vec3))
 
    #np.save("PullSthFromLeftToRight.npy",vec)

  input_ids4 = torch.tensor([tokenizer.encode("Push sth from right to left")])

  with torch.no_grad():
    last_hidden_states4 = model(input_ids4)[0][0][-1]
    print(last_hidden_states4.size())
    vec4 = last_hidden_states4.numpy()
    print(np.sum(vec4))
    #np.save("PushSthFromRightToLeft.npy",vec)

    print(np.sum(np.abs(vec3-vec4)))
