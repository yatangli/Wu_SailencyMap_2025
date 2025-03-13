from tkinter import messagebox

import datetime, h5py, math
import glob
import logging
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import tkinter as tk
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

def h5py_read(file_path,group_read='all',dataset_read='all'):    
    with h5py.File(file_path, "a") as f:
        data_dict = {}
        if group_read == 'all':
            for group in f:
                data_dict[group] = {}
                for dataset in f[group]:
                    data_dict[group][dataset] = f[group][dataset][()]
        elif dataset_read=='all':
            data_dict[group_read] = {}
            for dataset in f[group_read]:
                data_dict[group_read][dataset] = f[group_read][dataset][()]
        else:
            data_dict[group_read] = {}
            data_dict[group_read][dataset_read] = f[group_read][dataset_read][()]
    return data_dict

def h5py_write(file_path, data_dict,overwrite=False):
    '''
    data_dict['behavior']['pupil_size'] is a numpy array
    '''

    with h5py.File(file_path, "a") as f:
        for grp_name in data_dict:
            # if overwrite:
                # if grp_name in f:
                #     del f[grp_name]
            grp = f.require_group(grp_name)
            for dset_name in data_dict[grp_name]:
                data_name = '{}/{}'.format(grp_name,dset_name)
                # print(data_name)
                if overwrite:
                    if data_name in f:
                        del f[data_name]
                        print('overwrite {} in {}'.format(dset_name, grp_name))
                data = data_dict[grp_name][dset_name]
                if np.issctype(type(data)):
                    data = np.array(data)
                
                dset = grp.require_dataset(dset_name, data.shape, data.dtype)
                dset[()] = data

def min_idx_2d(arr):
    '''
    retuen the indexs of the min value of 2d array
    arr: 2d array, no nan in the array
    '''
    idx = np.zeros(2, dtype=int)
    idx[0] = np.nanargmin(np.min(arr, axis=1))
    idx[1] = np.nanargmin(np.min(arr, axis=0))
    return idx