import torch.nn as nn
import torch as th

class MultiColumn(nn.Module):
  def __init__(self, num_classes, conv_column, column_units, clf_layers=None):
    """
    - Example multi-column network
    - Useful when a video sample is too long and has to be split into 
      multiple clips
    - Processes 3D-CNN on each clip and averages resulting features across
      clips before passing it to classification(FC) layer.
   
    Args:
    - Input: Takes in a list of tensors each of size (batch_size, 3, sequence_length, W, H)
    - Returns: logits of size (batch_size, num_classes)
    """
    super(MultiColumn,self).__init__()
    self.num_classes = num_classes
    self.column_units = column_units
    self.conv_column = conv_column(column_units)
    self.clf_layers = clf_layers

    if not self.clf_layers:
      self.clf_layers = th.nn.Sequential(
                           nn.Linear(column_units, self.num_classes)
                        )

  def forward(self, inputs, get_features=False):
    outputs = []
    num_cols = len(inputs)
    for idx in range(num_cols):
      x = inputs[idx]
      x1 = self.conv_column(x)
      outputs.append(x1)

    outputs = th.stack(outputs).permute(1,0,2)
    outputs = th.squeeze(th.sum(outputs, 1), 1)
    avg_output = outputs / float(num_cols)
    outputs = self.clf_layers(avg_output)

    if get_features:
      return outputs, avg_output
    else:
      return outputs

if __name__ == "__main__":
  from model3D import Model
  num_classes = 174
  input_tensor = [th.autograd.Variable(th.rand(1,3,72,84,84)) for i in range(5)]
  print(input_tensor[0].size())
  model = MultiColumn(174, Model, 512)
  output = model(input_tensor)
