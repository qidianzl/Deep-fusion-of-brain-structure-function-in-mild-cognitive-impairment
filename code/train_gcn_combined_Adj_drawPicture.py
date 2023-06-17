import argparse
import time
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import shutil
import random
import pickle
from dataloader import get_loader, load_data, normlize_data
from model_gcn_combined_adj import GCN, GCN_BN

from utils import *

from tensorboardX import SummaryWriter
import pdb

# Device configuration
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    if os.path.exists('./txt/'):
    	shutil.rmtree('./txt/')
	os.makedirs('./txt/')


    # pdb.set_trace()
    # Build data loader
    all_data = load_data(args.data_root, args.data_type)
    data_mean = normlize_data(all_data)
    # pdb.set_trace()

    # torch.autograd.set_detect_anomaly(True)
    exp_prec = []
    for exp_num in range(10):
        f_acc = open('./txt/acc_avg_3_' + str(exp_num) + '.txt', 'a')
        f_model_epoch = open('./txt/model_epoch_' + str(exp_num) + '.txt', 'a')
        # Build the models
        model = GCN_BN(args.input_size, args.feat_size, args.feat_size * 2, 2, args.var,
                           args.M_row, args.M_col, 0.6)

        test_model = GCN_BN(args.input_size, args.feat_size, args.feat_size * 2, 2, args.var,
                                args.M_row, args.M_col, 0.6)
        # model = GCN_RNN_BN(args.input_size, args.feat_size, args.feat_size*2, args.hidden_size, 2, 0.6)
        # test_model = GCN_RNN_BN(args.input_size, args.feat_size, args.feat_size*2, args.hidden_size, 2, 0.6)
        model = model.to(device)
        test_model = test_model.to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
        # optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad is not False, model.parameters()), lr=args.learning_rate, weight_decay=1e-2)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=10)

        writer = SummaryWriter()
        best_prec = 0.0
        for epoch in range(args.num_epochs):
            train_data_loader = get_loader(args.data_root, all_data, data_mean, args.data_type, True, False,
                                           args.batch_size, num_workers=args.num_workers)

            val_data_loader = get_loader(args.data_root, all_data, data_mean, args.data_type, False, False,
                                         args.batch_size, num_workers=args.num_workers)

            test_data_loader = get_loader(args.data_root, all_data, data_mean, args.data_type, False, True,
                                      args.batch_size, num_workers=args.num_workers)
            # if epoch % 80 == 0 and epoch > args.batch_size0:
            #     adjust_learning_rate(optimizer, 0.1)

            # for name, p in model.named_parameters():
            #     skip = 30
            #     if name.startswith('combinedADJ.weight1') and epoch % skip == 0:
            #         p.requires_grad = False
            #     if name.startswith('combinedADJ.weight1') and epoch % skip != 0:
            #         p.requires_grad = True

            # for name, p in model.named_parameters():
            #     print "epoch:", epoch
            #     print name
            #     print (p.requires_grad)
            #
            # for p in optimizer.param_groups[0]['params']:
            #     print "epoch:", epoch
            #     print(p.requires_grad)
            #     print(type(p))

            train_acc = train(train_data_loader, model, criterion, optimizer, writer, epoch)
            prec = validate(val_data_loader, model, criterion, writer, epoch)
            test_acc = validate(test_data_loader, model, criterion, writer, epoch)
            f_acc.write(str(train_acc) + ',' + str(prec) + ',' + str(test_acc))
            f_acc.write('\n')
            scheduler.step(prec)
            # Save the model checkpoints
            if prec > best_prec:
                best_prec = prec
                torch.save(model.state_dict(), os.path.join(
                    args.model_path, 'miccai_rnn_gcn_64_1024_0.001_data2_model_best_exp-{}.ckpt'.format(exp_num)))
                f_model_epoch.write(epoch)
                f_model_epoch.write('\n')


        test_data_loader = get_loader(args.data_root, all_data, data_mean, args.data_type, False, True,
                                      1, num_workers=args.num_workers)
        f_acc.close()
        f_model_epoch.close()
        test_prec = test(test_data_loader, test_model, criterion, writer, exp_num)
        print ("Test Prec:", test_prec)
        exp_prec.append(test_prec)
        writer.close()

        del model
        del test_model
        del criterion
        del optimizer
        del scheduler
        del writer
    print(exp_prec)


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
    for i, (subject, feats, adjs, labels, pairwise_distance, fmri_Pearson) in enumerate(data_loader):
        data_time.update(time.time() - start)
        # pdb.set_trace()
        # Set mini-batch dataset
        feats = feats.to(device).float()
        adjs = adjs.to(device).float()
        labels = labels.to(device)
        pairwise_distance = pairwise_distance.to(device).float()
        fmri_Pearson = fmri_Pearson.to(device).float()

        # pdb.set_trace()
        # Forward, backward and optimize
        # for j in range(200):
        scores = model(feats, adjs, pairwise_distance, fmri_Pearson)
        # loss = criterion(torch.sigmoid(scores), onehot_labels)
        loss = criterion(scores, labels)

        model.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # for name, param in model.named_parameters():
        #     print name, ':', (param.data.size())
        # print"M:\n",(model.combinedADJ.weight1)
        # print"M max:",model.combinedADJ.weight1.max()
        # print"M min:",model.combinedADJ.weight1.min()
        # M = model.combinedADJ.weight1.detach().numpy()
        # rank = np.linalg.matrix_rank(M)
        # print"M rank:",rank
        #
        # print"beta:\n",(model.combinedADJ.weight2)

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
    print('Train Epoch: [{0}][{1}/{2}]\t'
          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(epoch, i, len(data_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                  top5=top5accs))
    # print"M:\n", (model.combinedADJ.weight1)
    # print"beta:\n", (model.combinedADJ.weight2)
    writer.add_histogram('gru_train_loss', losses.avg, epoch)
    writer.add_histogram('gru_train_acc', top5accs.avg, epoch)
    return top5accs.avg


def validate(data_loader, model, criterion, writer, epoch):
    # Evaluate the models
    model.eval()
    total_step = len(data_loader)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()

    start = time.time()
    for i, (subject, feats, adjs, labels, pairwise_distance, fmri_Pearson) in enumerate(data_loader):
        data_time.update(time.time() - start)
        # pdb.set_trace()
        # Set mini-batch dataset
        feats = feats.to(device).float()
        adjs = adjs.to(device).float()
        labels = labels.to(device)
        pairwise_distance = pairwise_distance.to(device).float()
        fmri_Pearson = fmri_Pearson.to(device).float()


        # Forward, backward and optimize
        with torch.no_grad():
            scores = model(feats, adjs, pairwise_distance, fmri_Pearson)
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
    print('Val Epoch: [{0}/{1}]\t'
          'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
          'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
          'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(i, len(data_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses,
                                                                  top5=top5accs))
    writer.add_histogram('gru_eval_loss', losses.avg, epoch)
    writer.add_histogram('gru_eval_acc', top5accs.avg, epoch)
    return top5accs.avg


def test(data_loader, model, criterion, writer, exp_num):
    # Evaluate the models
    if not os.path.exists('./results'):
        os.makedirs('./results')

    if not os.path.exists('./orig_adjs'):
        os.makedirs('./orig_adjs')

    if not os.path.exists('./final_adjs'):
        os.makedirs('./final_adjs')

    model.load_state_dict(torch.load(os.path.join(
        './models', 'miccai_rnn_gcn_64_1024_0.001_data2_model_best_exp-{}.ckpt'.format(exp_num))))
    model.eval()

    output_file = os.path.join(
        './results', 'miccai_rnn_gcn_64_1024_0.001_data2_results_exp-{}.txt'.format(exp_num))
    total_step = len(data_loader)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()

    outputs = []
    start = time.time()
    for i, (subject, feats, adjs, labels, pairwise_distance, fmri_Pearson) in enumerate(data_loader):
        data_time.update(time.time() - start)
        # pdb.set_trace()
        # Set mini-batch dataset
        feats = feats.to(device).float()
        adjs = adjs.to(device).float()
        labels = labels.to(device)
        pairwise_distance = pairwise_distance.to(device).float()
        fmri_Pearson = fmri_Pearson.to(device).float()


        # Forward, backward and optimize
        with torch.no_grad():
            scores, final_adj = model(feats, adjs, pairwise_distance, fmri_Pearson, isTest=True)
            loss = criterion(scores.unsqueeze(0), labels)
            result = F.softmax(scores, dim=0).cpu().numpy()
            # loss = criterion(torch.sigmoid(scores), onehot_labels)

        top5 = accuracy(scores.unsqueeze(0), labels, 1)
        # top5 = accuracy(torch.sigmoid(scores), labels, 1)
        losses.update(loss.item())
        top5accs.update(top5)
        batch_time.update(time.time() - start)

        start = time.time()
        result_str = ','.join([str(p) for p in list(result)])
        outputs.append(subject[0] + "," + result_str)

        adjs = adjs.squeeze().cpu().numpy()
        np.savetxt('./orig_adjs/' + subject[0][0:10] + '_orig_adj_' + str (exp_num) + '.txt', adjs, fmt='%.9f')

        final_adj = final_adj.squeeze().cpu().numpy()
        np.savetxt('./final_adjs/' + subject[0][0:10] + '_final-adj_' + str (exp_num) + '.txt', final_adj, fmt='%.9f')
        # writer.add_histogram('eval_loss', loss.item(), epoch * len(data_loader) + i)
        # writer.add_histogram('eval_acc', top5, epoch * len(data_loader) + i)

    with open(output_file, 'w') as fp:
        for subject_str in outputs:
            fp.write(subject_str)
            fp.write("\n")
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
    M = model.combinedADJ.weight1
    beta = model.combinedADJ.weight2
    retio = F.softmax(beta, dim=0)
    print("beta:", beta)
    print("retio:", retio)
    print("M:", M)
    return top5accs.avg


if __name__ == '__main__':
    # folder_path = '/Users/zl/Documents/Data/PycharmProgram/MICCAI/adni_data/data/Brain_lib/data_miccai'
    folder_path = '/data/lu_zhang/PycharmProgram/MICCAI/adni_data/data/Brain_lib/data_miccai'

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/', help='path for saving trained models')
    parser.add_argument('--data_root', type=str, default=folder_path, help='path for image data')
    parser.add_argument('--log_step', type=int, default=10, help='step size for prining log info')
    parser.add_argument('--save_step', type=int, default=1, help='step size for saving trained models')
    # Model parameters
    parser.add_argument('--input_size', type=int, default=148, help='dimension of word embedding vectors')
    parser.add_argument('--feat_size', type=int, default=148, help='dimension of lstm hidden states')
    parser.add_argument('--data_type', type=int, default=2, help='data type of input')
    # math parameters
    parser.add_argument('--M_row', type=int, default=45, help='the row of M matrix, equal to the time of fmri signal')
    parser.add_argument('--M_col', type=int, default=45, help='the col of M matrix, equal to the time of fmri signal')
    parser.add_argument('--var', type=int, default=2, help='the var of kernal function')

    parser.add_argument('--num_epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    args = parser.parse_args()
    print(args)
    main(args)