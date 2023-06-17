from __future__ import print_function

import torch.utils.data as data
import os
import random
import os.path
import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import pearsonr

from sampler import BalancedBatchSampler

import pdb


def label_dict():
    subject_class = {}
    CN_file = './CN_used.txt'
    MCI_folder = './MCI_used.txt'
    # pdb.set_trace()
    with open(MCI_folder, 'r') as f:
        for subj in f.readlines():
            subj = subj.replace("\n", "")
            subject_class[subj] = 0

    with open(CN_file, 'r') as f:
        for subj in f.readlines():
            subj = subj.replace("\n", "")
            subject_class[subj] = 1

    return subject_class


def laplace(A):
    D = np.zeros(shape=A.shape)
    for i in range(D.shape[0]):
        D[i][i] = sum(A[i])

    L1 = D - A
    L2 = np.matmul(np.matmul(sqrtm(np.linalg.inv(D)), L1), sqrtm(np.linalg.inv(D)))
    L3 = np.matmul(np.linalg.inv(D), L1)

    return L2


def load_data(data_path, data_type):
    # pdb.set_trace()
    data = []
    timewindow = 4
    localtime = 45
    SubjID_list = [x for x in os.listdir(data_path) if not x.startswith('.')]
    data_num = len(SubjID_list) * timewindow

    # pdb.set_trace()
    for suj_num in range(0, data_num, timewindow):
        subject_index = int(suj_num / timewindow)
        subject = SubjID_list[subject_index]

        subject_class = label_dict()
        if subject not in subject_class.keys():
            continue
        # pdb.set_trace()
        label = subject_class[subject]

        # adj matrix part
        fiber_matrix_path = data_path + '/' + subject + '/' + 'nonzero_common_fiber_matrix.npy'

        if data_type == 0:
            fmri_matrix_folder_path = data_path + '/' + subject + '/' + 'nonzero_fmri_average_signal'
        elif data_type == 1:
            fmri_matrix_folder_path = data_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized1'
        else:
            fmri_matrix_folder_path = data_path + '/' + subject + '/' + 'nonzero_fmri_average_signal_normalized2'

        for time_1 in range(timewindow):
            timepoint = time_1 * localtime
            fmri_matrix_prefix = fmri_matrix_folder_path + '/' + 'raw_fmri_feature_matrix_'
            data.append((subject + '_' + str(label), fiber_matrix_path, fmri_matrix_prefix, timepoint, label))
    random.shuffle(data)
    return data


def get_feats(fmri_matrix_prefix, start_time):
    localtime = 45
    feats = []
    for time in range(localtime):
        timepoint = start_time + time
        # print 'timewindow', time_1
        # print 'localtime', time_2
        # print 'timepoint', timepoint
        fmri_matrix_path = fmri_matrix_prefix + str(timepoint) + '.npy'
        # pdb.set_trace()
        # print fmri_matrix_path
        fmri_matrix = np.load(fmri_matrix_path)
        feats.append(fmri_matrix)

    feats = np.stack(feats)
    return feats


def normlize_data(data):
    #pdb.set_trace()
    eps = 1e-9
    all_feats = []
    all_adjs = []
    all_pair_dist = []
    all_fmri_Pearson = []
    for index in range(len(data)):
        print(index)
        subject, fiber_matrix_path, fmri_matrix_prefix, timepoint, label = data[index]

        fiber_matrix = np.load(fiber_matrix_path)
        fiber_matrix = adj_matrix_normlize(fiber_matrix)
        fmri_matrix = get_feats(fmri_matrix_prefix, timepoint)
        pair_dist = pairwise_distance(fmri_matrix)
        fmri_Pearson = corelation_matrix(fmri_matrix)


        all_feats.append(fmri_matrix)
        all_adjs.append(fiber_matrix)
        all_pair_dist.append(pair_dist)
        all_fmri_Pearson.append(fmri_Pearson)

    all_feats = np.stack(all_feats)
    all_adjs = np.stack(all_adjs)
    all_pair_dist = np.stack(all_pair_dist)
    all_fmri_Pearson = np.stack(all_fmri_Pearson)

    # pdb.set_trace()
    feat_mean = all_feats.mean((0, 1), keepdims=True).squeeze(0)
    feat_std = all_feats.std((0, 1), keepdims=True).squeeze(0)

    adj_mean = all_adjs.mean((0, 1, 2), keepdims=True).squeeze(0)
    adj_std = all_adjs.std((0, 1, 2), keepdims=True).squeeze(0)

    pair_dist_mean = all_pair_dist.mean((0, 3), keepdims=True).squeeze(0)
    pair_dist_std = all_pair_dist.std((0, 3), keepdims=True).squeeze(0)
    fmri_Pearson_mean = all_fmri_Pearson.mean((0, 1, 2), keepdims=True).squeeze(0)
    fmri_Pearson_std = all_fmri_Pearson.std((0, 1, 2), keepdims=True).squeeze(0)


    # print (all_feats.shape)
    # print (feat_mean.shape)
    # print (all_adjs.shape)
    # print (adj_mean.shape)
    # print (pair_dist_mean.shape)
    # print (pair_dist_std.shape)

    # print (pair_dist_mean)
    # print (pair_dist_std)
    # pdb.set_trace()

    return (torch.from_numpy(feat_mean) + eps, torch.from_numpy(feat_std) + eps, torch.from_numpy(adj_mean) + eps,
            torch.from_numpy(adj_std) + eps, torch.from_numpy(pair_dist_mean) + eps,
            torch.from_numpy(pair_dist_std) + eps, torch.from_numpy(fmri_Pearson_mean) + eps, torch.from_numpy(fmri_Pearson_std) + eps)


def adj_matrix_normlize(adj):
    adj_norm = adj + 1
    adj_norm = np.log10(adj_norm)
    return adj_norm


def pairwise_distance(feats):
    region_num = feats.shape[1]
    time = feats.shape[0]

    pair_dist = np.zeros((region_num, region_num, time))
    for i in range(region_num):
        for j in range(region_num):
            # pdb.set_trace()
            dist = feats[:, i] - feats[:, j]
            pair_dist[i, j] = dist.squeeze()

    return pair_dist


def signal_corelation(signal1, signal2):
    if np.all(signal1 == 0) or np.all(signal2 == 0):
        pcc = 0
    else:
        pcc, p_value = pearsonr(signal1, signal2)
    return pcc


def corelation_matrix(fmri_matrix):
    regionSize = fmri_matrix.shape[1]
    # print (regionSize)
    fmri_Pearson = np.zeros(shape=(regionSize, regionSize))
    for region1 in range(regionSize):
        for region2 in range(regionSize):
            fmri_Pearson[region1, region2] = signal_corelation(fmri_matrix[:, region1], fmri_matrix[:, region2])
    # print (fmri_Pearson)
    # pdb.set_trace()

    return fmri_Pearson


class MICCAI(data.Dataset):
    def __init__(self, data_path, all_data, data_mean, data_type=1, train=True, test=False):
        self.data_path = data_path
        self.data_type = data_type
        self.train = train  # training set or val set
        self.test = test
        self.feat_mean, self.feat_std, self.adj_mean, self.adj_std, self.pair_dist_mean, self.pair_dist_std, self.fmri_Pearson_mean, self.fmri_Pearson_std = data_mean

        # pdb.set_trace()
        if self.train:
            self.data = all_data[:330]
            # self.data = all_data[:151]
        elif not test:
            self.data = all_data[330:396]
            # self.data = all_data[151:]
        else:
            self.data = all_data[396:]

        random.shuffle(self.data)

    def _get_feats(self, fmri_matrix_prefix, start_time):
        localtime = 45
        feats = []
        for time in range(localtime):
            timepoint = start_time + time
            # print 'timewindow', time_1
            # print 'localtime', time_2
            # print 'timepoint', timepoint
            fmri_matrix_path = fmri_matrix_prefix + str(timepoint) + '.npy'
            # pdb.set_trace()
            # print fmri_matrix_path
            fmri_matrix = np.load(fmri_matrix_path)
            feats.append(fmri_matrix)

        feats = np.stack(feats)
        return feats

    def __getitem__(self, index):
        subject, fiber_matrix_path, fmri_matrix_prefix, timepoint, label = self.data[index]

        fiber_matrix = np.load(fiber_matrix_path)
        fiber_matrix = adj_matrix_normlize(fiber_matrix)
        # l2 = laplace(fiber_matrix)
        # fiber_matrix = l2

        adj_matrix = torch.from_numpy(fiber_matrix)
        adj_matrix = (adj_matrix - self.adj_mean) / self.adj_std

        fmri_matrix = self._get_feats(fmri_matrix_prefix, timepoint)

        fmri_Pearson = corelation_matrix(fmri_matrix)
        fmri_Pearson = torch.from_numpy(fmri_Pearson)
        fmri_Pearson = (fmri_Pearson - self.fmri_Pearson_mean) / self.fmri_Pearson_std


        pair_dist = pairwise_distance(fmri_matrix)
        pair_dist = torch.from_numpy(pair_dist)
        pair_dist = (pair_dist - self.pair_dist_mean) / self.pair_dist_std

        fmri_matrix = torch.from_numpy(fmri_matrix)
        fmri_matrix = (fmri_matrix - self.feat_mean) / self.feat_std

        # print (fmri_matrix.shape)
        # print (adj_matrix.shape)
        # print (pair_dist.shape)
        # print (fmri_Pearson.shape)
        # pdb.set_trace()

        return subject, fmri_matrix, adj_matrix, label, pair_dist, fmri_Pearson

    def debug_getitem__(self, index=0):
        pdb.set_trace()
        subject, fiber_matrix_path, fmri_matrix_prefix, timepoint, label = self.data[index]

        fiber_matrix = np.load(fiber_matrix_path)
        # D, l1, l2, l3 = laplace(fiber_matrix)
        # location = np.where(adj == adj.max())
        # print adj[location[0][0]][location[1][0]]
        # adj = adj / adj.max()
        adj_matrix = torch.from_numpy(fiber_matrix)
        adj_matrix = (adj_matrix - self.adj_mean) / self.adj_std

        fmri_matrix = self._get_feats(fmri_matrix_prefix, timepoint)
        fmri_matrix = torch.from_numpy(fmri_matrix)

        fmri_matrix = (fmri_matrix - self.feat_mean) / self.feat_std

        return subject, fmri_matrix, adj_matrix, label

    def __len__(self):
        return len(self.data)


def get_loader(data_path, data_list, data_mean, data_type, training, test, batch_size=16, num_workers=4):
    dataset = MICCAI(data_path, data_list, data_mean, data_type, training, test)

    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              sampler=BalancedBatchSampler(dataset),
                                              batch_size=batch_size,
                                              num_workers=num_workers)

    return data_loader


if __name__ == '__main__':
    data_path = './data_miccai'

    all_data = load_data(data_path, 2)
    data_mean = normlize_data(all_data)

    dataset = MICCAI(data_path, all_data, data_mean, 2)
    for i in range(len(dataset)):
        x, y, z = dataset.debug_getitem__(i)