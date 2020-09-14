import os
import sys
import time
import signal
import importlib

import torch
import torch.nn as nn
import numpy as np

from utils import *
from callbacks import (PlotLearning, AverageMeter)
from models.multi_column import MultiColumn
import torchvision
from transforms_video import *

# load configurations
args = load_args()
config = load_json_config(args.config)

# set column model
file_name = config['conv_model']
cnn_def = importlib.import_module("{}".format(file_name))

# setup device - CPU or GPU
device, device_ids = setup_cuda_devices(args)
print("> Using device: {}".format(device.type))
print("> Active GPU ids:{}".format(device_ids))

best_loss = float('Inf')

if config["input_mode"] == "av":
  from data_loader_av import VideoFolder
elif config["input_mode"] == "skvideo":
  from data_loader_skvideo import VideoFolder
else:
  raise ValueError("Please provide3 a valid input mode")

def main():
  global args, best_loss

  # set run output folder
  model_name = config["model_name"]
  output_dir = config["output_dir"]
  save_dir = os.path.join(output_dir,model_name)
  print(" > Output folder for this run -- {}".format(save_dir))
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir,'plots'))

  # assign Ctrl+C signal handler
  signal.signal(Signal.SIGINT, ExperimentalRunCleaner(save_dir))

  # create model
  print(" > Creating model.... !")
  model = MultiColumn(config['num_classes'], cnn_def.Model, int(config["column_units"]))

  # multi GPU setting
  model = torch.nn.DataParallel(model, device_ids).to(device)

  # optionally resume from a checkpoint
  checkpoint_path = os.path.join(config['output_dir'],
                                 config['model_name'],
                                 'model_best.path.tar')

  if args.resume:
    if os.path.isfile(checkpoint_path):
      print("> Loading checkpoint '{}'".format(args.resume))
      checkpoint = torch.load(checkpoint_path)
      args.start_epoch = checkpoint['epoch']
      best_loss = checkpoint['best_loss']
      model.load_state_dict(checkpoint['state_dict'])
      print("> Loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))
    else:
      print("!#! No checkpoint found at '{}'".format(checkpoint_path))

  # define augmentation pipeline
  upscale_size_train = int(config['input_spatial_size'] * config['upscale_factor_train'])
  upscale_size_eval  = int(config['input_spatial_size'] * config['upscale_factor_eval'])

  # Random crop videos during trainig
  transform_train_pre = ComposeMix([
          [RandomRotationVideo(15), "vid"],
          [Scale(upscale_size_train),"img"],
          [RandomCropVideo(config['input_spatial_size']),"vid"],
          ])

  # Center crop videos during evaluation
  transform_eval_pre = ComposeMix([
          [Scale(upscale_size_eval),"img"],
          [torchvision.transforms.ToPILImage(),"img"],
          [torchvision.transforms.CenteCrop(config['input_spatial_size']),"img"],
        ])

  # Transform common to train and eval sets and applied after "pre" transforms
  transform_post = ComposeMix([
          [torchvision.transforms.ToTensor(),"img"],
          [torchvision.transforms.Normalize(
                  mean=[0.485,0.456,0.406],
                  std=[0.229,0.224,0.225]),"img"]
           ])

  train_data = VideoFolder(root=config['data_folder'],
                           json_file_input=config['json_data_train'],
                           json_file_labels=config['json_file_labels'],
                           clip_size=config['clip_size'],
                           nclips=config['nclips_train'],
                           step_size=config['step_size_train'],
                           is_val=False,
                           transform_pre=transform_pre,
                           transform_post=transform_post,
                           augmentation_mappings_json=config['augmentation_mappings_json'],
                           augmentation_types_todo=config['augmentation_types_todo'],
                           get_item_id=False
                           )
   
  print("> Using {} processes for data loader.".format(config["num_workers"]))

  train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=config['batch_size'], shuffle=True,
    num_workers=config['num_workers'], pin_memeory=True,
    drop_last=True)

  val_data = VideoFolder(root=config['data_folder'],
                         json_file_input=config['json_data_val'],
                         json_file_labels=config['json_file_labels'],
                         clip_size=config['clip_size'],
                         nclips=config['nclips_val'],
                         step_size=config['step_size_val'],
                         is_val=True,
                         transform_pre=transform_eval_pre,
                         transform_post=transform_post,
                         get_item_id=True,)

  val_loader = torch.utils.data.DataLoader(
     val_data,
     batch_size=config['batch_size'], shuffle=False,
     num_workers=config['num_workers'], pin_memory=True,
     drop_last=False)

  print(" >Number of dataset classes: {}".format(len(train_data.classes)))

  # define loss function (criterion)
  criterion = nn.CrossEntropyLoss().to(device)

  # define optimizer
  lr = config["lr"]
  last_lr = config["last_lr"]
  momentum = config["momentum"]
  weight_decay = config['weight_decay']
  optimizer = torch.optim.SGD(model.parameters(),lr,momentum=momentum,weight_decay=weight_decay)

  # set callbacks
  plotter = PlotLearning(os.path.join(
        save_dir, "plots"), config["num_classes"])
  lr_decayer = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 'min', factor=0.5, patience=2, verbose=True)
  val_loss = float('Inf') 

  # set end condition by num epochs
  num_epochs = int(config["num_epochs"])
  if num_epochs == -1:
    num_epochs = 999999

  print(" > Training is getting started...")
  print(" > Training takes {} epochs.".format(num_epochs))
  start_epoch = args.start_epoch if args.resume else 0

  for epoch in range(start_epoch, num_epochs):
    lrs = [params['lr'] for params in optimizer.param_groups]
    print("> Current LR(s) -- {}".format(lrs))
    if np.max(lr) < last_lr and last_lr > 0:
      print("Training is DONE by learning rate {}".format(last_lr))
      sys.exit(1)

    # train for one epoch
    train_loss, train_top1, train_top5 = train(
         train_loader, model, criterion, optimizer, epoch)
 
    # evaluate on evaluation set
    val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)
  
    # set learing rate
    lr_decayer.step(val_loss, epoch)

    # plot learning
    plotter_dict = {}
    plotter_dict['loss'] = train_loss
    plotter_dict['val_loss'] = val_loss
    plotter_dict['acc'] = train_top1 / 100
    plotter_dict['val_acc'] = val_top1 / 100
    plotter_dict['learning_rate'] = lr
    plotter.plot(plotter_dict)

    print(" > Validation loss after epoch {} = {}".format(epoch, val_loss))

    # remember best loss and save the checkpoint
    is_best = val_loss < best_loss
    best_loss = min(val_loss, best_loss)
    save_checkpoint({
            'epoch': epoch + 1,
            'arch': "Conv4Col",
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
        }, is_best, config)

def train(train_loader, model, criterion, optimizer, epoch):
   batch_time = AverageMeter()
   data_time = AverageMeter()
   losses = AverageMeter()
   top1 = AverageMeter()
   top5 = AverageMeter()

   # switch to train mode
   model.train()

   end = time.time()
 
   for i, (input, target) in enumerate(train_loader):
     # measure data loading time
     data_time.update(time.time() - end)

     if config['nclips_train'] > 1:
       input_var = list(input.split(config['clip_size'],2))
       for idx, inp in enumerate(input_var):
         input_var[idx] = inp.to(device)
     else:
       input_var = [input.to(device)]
     target = target.to(device)

