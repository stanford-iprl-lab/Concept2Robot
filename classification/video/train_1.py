import av
import torch
import numpy as np
import os
import sys
import importlib
import signal
import time

import torch
import torch.nn as nn

from utils import *
from callbacks import (PlotLearning, AverageMeter)
from models.multi_column import MultiColumn


from data_parser import WebmDataset
from data_augmentor import Augmentor
import torchvision
from transforms_video import *
from utils import save_images_for_debug
from data_loader_av import VideoFolder


# load configurations
args = load_args()
config = load_json_config(args.config)

# set column model
file_name = config['conv_model']
print("file_name",file_name)
cnn_def = importlib.import_module("{}".format(file_name))

# set up device - CPU or GPU
device, device_ids = setup_cuda_devices(args)
print("> Using device:{}".format(device.type))
print("> Active GPU ids:{}".format(device_ids))

best_loss = float('Inf')

if config["input_mode"] == "av":
  from data_loader_av import VideoFolder
elif config["input_mode"] == "skvideo":
  from data_loader_skvideo import VideoFolder
else:
  raise	ValueError("Please provide a valid input mode.")

def main():
  global args, best_loss

  # set run output folder
  model_name = config["model_name"]
  output_dir = config["output_dir"]
  save_dir = os.path.join(output_dir, model_name)
  print(" > Output folder for this run -- {}".format(save_dir))
  if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    os.makedirs(os.path.join(save_dir,'plots'))

  # assign Ctrl+C singal hanlder
  signal.signal(signal.SIGINT, ExperimentalRunCleaner(save_dir))

  # create model
  print(" > Creating model...!")
  model = MultiColumn(config['num_classes'], cnn_def.Model, int(config["column_units"]))

  # multi GPU setting
  model = torch.nn.DataParallel(model, device_ids).to(device)

  # optinally resume from a checkpoint

  # define augmentation pipeline
  upscale_size_train = int(config['input_spatial_size'] * config["upscale_factor_train"])
  upscale_size_eval =  int(config['input_spatial_size'] * config["upscale_factor_eval"])

  # Random crop videos during training
  transform_train_pre = ComposeMix([
      [RandomRotationVideo(15),"vid"],
      [Scale(upscale_size_train),"img"],
      [RandomCropVideo(config["input_spatial_size"]),"vid"],
    ])

  # Center crop videos during evaluation
  transform_eval_pre = ComposeMix([
      [Scale(upscale_size_eval),"img"],
      [torchvision.transforms.ToPILImage(),"img"],
      [torchvision.transforms.CenterCrop(config['input_spatial_size']),"img"],
   ])

  # Transforms common to train and eval sets and applied after "pre" transforms
  transform_post = ComposeMix([
     [torchvision.transforms.ToTensor(),"img"],
     [torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # default values for imagenet
        std=[0.229, 0.224, 0.225]), "img"]
     ])

  train_data = VideoFolder(root=config["data_folder"],
                           json_file_input=config["json_data_train"],
                           json_file_labels=config["json_file_labels"],
                           clip_size=config['clip_size'],
                           nclips=config['nclips_train'],
                           step_size=config['step_size_train'],
                           is_val=False,
                           transform_pre=transform_train_pre,
                           transform_post=transform_post,
                           get_item_id=False,
                          )

  print(" > Using {} processes for data loader.".format(
        config["num_workers"]))

  train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=config['batch_size'], shuffle=True,
    num_workers=config['num_workers'], pin_memory=True,
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
                           get_item_id=True,
                           )

  val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config['batch_size'], shuffle=False,
        num_workers=1, pin_memory=True,
        drop_last=False)


  # define loss function (criterion)
  criterion = nn.CrossEntropyLoss().to(device)

  # define optimizer
  lr = config['lr']
  last_lr = config['last_lr']
  momentum = config['momentum']
  weight_decay = config['weight_decay']
  optimizer = torch.optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=weight_decay)

  if args.eval_only:
    validate(test_loader, model, criterion, train_data.classes_dict)
    print(" > Evalution Done!")
    return

  # set callbacks
  plotter = PlotLearning(os.path.join(save_dir,"plots"),config["num_classes"])
  lr_decayer = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,'min',factor=0.5,patience=2,verbose=True)
  var_loss = float('Inf')

  # set and condition by num epochs
  num_epochs = int(config["num_epochs"]) 
  if num_epochs == -1:
    num_epochs = 9999999

  print(" > Training is getting started...")
  print(" > Training takes {} epochs.".format(num_epochs))
  start_epoch = args.start_epoch if args.resume else 0

  for epoch in range(start_epoch, num_epochs):
    lrs = [params['lr'] for params in optimizer.param_groups]
    print(" > Current LR(s) -- {}".format(lrs))

    if np.max(lr) < last_lr and last_lr > 0:
      print("> Training is DONE by learning rate {}".format(last_lr))
      sys.exit(1) 

    # train for one epoch
    train_loss, train_top1, train_top5 = train(
      train_loader, model, criterion, optimizer, epoch)

    # evaluate on validation set
    val_loss, val_top1, val_top5 = validate(val_loader, model, criterion)
    #val_loss, val_top1, val_top5 = 0,0,0
    # set learning rate
    lr_decayer.step(val_loss, epoch)

    # plot learning
    plotter_dict = {}
    plotter_dict['loss'] = train_loss
    plotter_dict['val_loss'] = val_loss
    plotter_dict['acc'] = train_top1 / 100
    plotter_dict['val_acc'] = val_top1 / 100
    plotter_dict['learning_rate']  = lr
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
    #measure data loading time
    data_time.update(time.time() - end)

    if config['nclips_train'] > 1:
      input_var = list(input.split(config['clip_size'], 2))
      for idx, inp in enumerate(input_var):
        input_var[idx] = inp.to(device)
    else:
      input_var = [input.to(device)]

    target = target.to(device)

    model.zero_grad()
 
    # compute output and loss
    output = model(input_var)
    loss = criterion(output, target)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1,5))
    losses.update(loss.item(),input.size(0))
    top1.update(prec1.item(), input.size(0))
    top5.update(prec5.item(), input.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()

    if i % config["print_freq"] == 0:
      print('Epoch: [{0}][{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
         epoch, i, len(train_loader), batch_time=batch_time,
         data_time=data_time, loss=losses, top1=top1, top5=top5))
 
    return losses.avg, top1.avg, top5.avg

def validate(val_loader,model,criterion,class_to_idx=None):
  batch_time = AverageMeter()
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()
 
  logits_matrix = []
  features_matrix = []
  targets_list = []
  item_id_list = []

  end = time.time()
  
  with torch.no_grad():
    for i, (input, target, item_id) in enumerate(val_loader):
      if config['nclips_val'] > 1:
        input_var = list(input.split(config['clip_size'],2))
        for idx, inp in enumerate(input_var):
          input_var[idx] = inp.to(device)
      else:
        input_var = [input.to(device)]
 
      target = target.to(device)

      # compute output and loss
      output, features = model(input_var, config['save_features'])
      loss = criterion(output, target)

      if args.eval_only:
        logits_matrix.append(output.cpu().data.numpy())
        features_matrix.append(features.cpu(),data.numpy())
        targets_list.append(target.cpu().numpy())
        item_id_list.append(item_id)

      # measure accuracy and record loss
      prec1, prec5 = accuracy(output.detach().cpu(), target.detach().cpu(), topk=(1,5))
      losses.update(loss.item(), input.size(0))
      top1.update(prec1.item(), input.size(0))
      top5.update(prec5.item(), input.size(0))

      # measure elapsed time
      batch_time.update(time.time() - end)
      end = time.time()

      if i % config["print_freq"] == 0:
        print('Test: [{0}/{1}]\t'
              'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                  i, len(val_loader), batch_time=batch_time, loss=losses,
             top1=top1, top5=top5))

      print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
           .format(top1=top1, top5=top5))

      if args.eval_only:
        logits_matrix = np.concatenate(logits_matrix)
        features_matrix = np.concatenate(features_matrix)
        targets_list = np.concatenate(targets_list)
        item_id_list = np.concatenate(item_id_list)
        print(logits_matrix.shape, targets_list.shape, item_id_list.shape)
        save_results(logits_matrix, features_matrix, targets_list,
                     item_id_list, class_to_idx, config)
        get_submission(logits_matrix, item_id_list, class_to_idx, config)
  return losses.avg, top1.avg, top5.avg
 

if __name__ == '__main__':
  main()
