import argparse
import time
import torch
import torch.nn as nn
import numpy as np
import os
import random
import pickle
from dataloader import get_loader, load_data, normlize_data
from models_lstm import GCN_RNN, GCN_RNN_BN
from utils import *

from tensorboardX import SummaryWriter
import pdb

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # pdb.set_trace()
    # Build data loader
    all_data = load_data(args.data_root, args.data_type)
    data_mean = normlize_data(all_data)

    exp_prec = []
    for exp_num in range(10):
        # Build the models
        model = GCN_RNN(args.input_size, args.feat_size, args.feat_size*2, args.hidden_size, 2, 0.6)
        test_model = GCN_RNN(args.input_size, args.feat_size, args.feat_size*2, args.hidden_size, 2, 0.6)
        # model = GCN_RNN_BN(args.input_size, args.feat_size, args.feat_size*2, args.hidden_size, 2, 0.6)
        # test_model = GCN_RNN_BN(args.input_size, args.feat_size, args.feat_size*2, args.hidden_size, 2, 0.6)
        model = model.to(device)
        test_model = test_model.to(device)
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

        writer = SummaryWriter()
        best_prec = 0.0
        for epoch in range(args.num_epochs):
            train_data_loader = get_loader(args.data_root, all_data, data_mean, args.data_type, True, False,
                                 args.batch_size, num_workers=args.num_workers)

            val_data_loader = get_loader(args.data_root, all_data, data_mean, args.data_type, False, False,
                                     args.batch_size, num_workers=args.num_workers)
            # if epoch % 80 == 0 and epoch > 0:
            #     adjust_learning_rate(optimizer, 0.1)
            train(train_data_loader, model, criterion, optimizer, writer, epoch)
            prec = validate(val_data_loader, model, criterion, writer, epoch)
            scheduler.step(prec)
            # Save the model checkpoints
            if prec > best_prec:
                best_prec = prec
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'miccai_lstm_bn_64_1024_0.001_data2_model_best_exp-{}.ckpt'.format(exp_num)))

        test_data_loader = get_loader(args.data_root, all_data, data_mean, args.data_type, False, True,
                                     args.batch_size, num_workers=args.num_workers)
        test_prec = test(test_data_loader, test_model, criterion, writer, exp_num)
        print "Test Prec:", test_prec
        exp_prec.append(test_prec)
        writer.close()

        del model
        del test_model
        del criterion
        del optimizer
        del scheduler
        del writer
    print exp_prec

def train(data_loader, model, criterion, optimizer, writer, epoch):
    # Train the models
    # pdb.set_trace()
    model.train()
    total_step = len(data_loader)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy

    start = time.time()
    for i, (subject, feats, adjs, labels) in enumerate(data_loader):
        data_time.update(time.time() - start)
        # pdb.set_trace()
        # Set mini-batch dataset
        feats = feats.to(device).float()
        adjs = adjs.to(device).float()
        labels = labels.to(device)

        # pdb.set_trace()
        # Forward, backward and optimize
        # for j in range(200):
        scores = model(feats, adjs)
        # loss = criterion(torch.sigmoid(scores), onehot_labels)
        loss = criterion(scores, labels)
        
        model.zero_grad()
        loss.backward()
        optimizer.step()

        # top5 = accuracy(torch.sigmoid(scores), labels, 1)
        top5 = accuracy(scores, labels, 1)
        losses.update(loss.item())
        top5accs.update(top5)
        batch_time.update(time.time() - start)

        start = time.time()

        # writer.add_histogram('train_loss', loss.item(), epoch * len(data_loader) + i)
        # writer.add_histogram('train_acc', top5, epoch * len(data_loader) + i)

        # Print log info
        # if i % args.log_step == 0:
        #     print('Train Epoch: [{0}][{1}/{2}]\t'
        #           'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(data_loader),
        #                                                                   batch_time=batch_time,
        #                                                                   data_time=data_time, loss=losses,
        #                                                                   top5=top5accs))
    writer.add_histogram('lstm_train_loss', losses.avg, epoch)
    writer.add_histogram('lstm_train_acc', top5accs.avg, epoch)

def validate(data_loader, model, criterion, writer, epoch):
    # Evaluate the models
    model.eval()
    total_step = len(data_loader)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()

    start = time.time()
    for i, (subject, feats, adjs, labels) in enumerate(data_loader):
        data_time.update(time.time() - start)
        # pdb.set_trace()
        # Set mini-batch dataset
        feats = feats.to(device).float()
        adjs = adjs.to(device).float()
        labels = labels.to(device)

        # Forward, backward and optimize
        with torch.no_grad():
            scores = model(feats, adjs)
            loss = criterion(scores, labels)
            # loss = criterion(torch.sigmoid(scores), onehot_labels)

        top5 = accuracy(scores, labels, 1)
        # top5 = accuracy(torch.sigmoid(scores), labels, 1)
        losses.update(loss.item())
        top5accs.update(top5)
        batch_time.update(time.time() - start)

        start = time.time()

        # writer.add_histogram('eval_loss', loss.item(), epoch * len(data_loader) + i)
        # writer.add_histogram('eval_acc', top5, epoch * len(data_loader) + i)

        # Print log info
        # if i % args.log_step == 0:
    # print('Val Epoch: [{0}/{1}]\t'
    #       'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
    #       'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
    #       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
    #       'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(data_loader),
    #                                                               batch_time=batch_time,
    #                                                               data_time=data_time, loss=losses,
    #                                                               top5=top5accs))
    writer.add_histogram('lstm_eval_loss', losses.avg, epoch)
    writer.add_histogram('lstm_eval_acc', top5accs.avg, epoch)
    return top5accs.avg

def test(data_loader, model, criterion, writer, exp_num):
    # Evaluate the models
    model.load_state_dict(torch.load(os.path.join(
                './models', 'miccai_lstm_bn_64_1024_0.001_data2_model_best_exp-{}.ckpt'.format(exp_num))))
    model.eval()
    total_step = len(data_loader)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()

    start = time.time()
    for i, (subject, feats, adjs, labels) in enumerate(data_loader):
        data_time.update(time.time() - start)
        # pdb.set_trace()
        # Set mini-batch dataset
        feats = feats.to(device).float()
        adjs = adjs.to(device).float()
        labels = labels.to(device)

        # Forward, backward and optimize
        with torch.no_grad():
            scores = model(feats, adjs)
            loss = criterion(scores, labels)
            # loss = criterion(torch.sigmoid(scores), onehot_labels)

        top5 = accuracy(scores, labels, 1)
        # top5 = accuracy(torch.sigmoid(scores), labels, 1)
        losses.update(loss.item())
        top5accs.update(top5)
        batch_time.update(time.time() - start)

        start = time.time()

        # writer.add_histogram('eval_loss', loss.item(), epoch * len(data_loader) + i)
        # writer.add_histogram('eval_acc', top5, epoch * len(data_loader) + i)

        # Print log info
        # if i % args.log_step == 0:
    print('Test Epoch: [{0}/{1}]\t'
          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(data_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                  top5=top5accs))
    return top5accs.avg


if __name__ == '__main__':
    folder_path = '/data/lu_zhang/PycharmProgram/MICCAI/adni_data/data/Brain_lib/data_miccai'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--data_root', type=str, default=folder_path, help='path for image data')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1, help='step size for saving trained models')
    # Model parameters
    parser.add_argument('--input_size', type=int, default=1, help='dimension of word embedding vectors')
    parser.add_argument('--feat_size', type=int, default=32, help='dimension of lstm hidden states')
    parser.add_argument('--hidden_size', type=int, default=1024, help='dimension of lstm hidden states')
    parser.add_argument('--data_type', type=int, default=2, help='data type of input')

    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)