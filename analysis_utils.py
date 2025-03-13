#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 09:35:24 2023

@author: yatangli
"""
import h5py
import glob
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math, datetime, os, scipy

from matplotlib.ticker import FormatStrFormatter
from other_utils import h5py_write, h5py_read, min_idx_2d
from scipy.optimize import curve_fit
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
import pandas as pd

#%%

def adjacent_values(vals, q1, q3):
    upper_adjacent_value = q3 + (q3 - q1) * 1.5
    upper_adjacent_value = np.clip(upper_adjacent_value, q3, vals[-1])

    lower_adjacent_value = q1 - (q3 - q1) * 1.5
    lower_adjacent_value = np.clip(lower_adjacent_value, vals[0], q1)
    return lower_adjacent_value, upper_adjacent_value

def f(x, a, b, k):
    return a - b*np.exp(-k * x)
    
def binned_mean(distances, data, bar_color):
    from scipy.optimize import curve_fit

    fontsize = 20
    _idx_not_nan = np.logical_not(np.isnan(data))
    bin_size = 10
    right_lim = 260
    bins = np.arange(0, right_lim, bin_size)

    fig, ax = plt.subplots(figsize=(5, 2))
    means, edges, _= scipy.stats.binned_statistic(distances[_idx_not_nan],
        data[_idx_not_nan], statistic='mean', bins=bins)
    facecolor = mcolors.to_rgb(bar_color)
    facecolor_alpha = (facecolor[0], facecolor[1], facecolor[2], 0.5)
    ax.bar(x=edges[:-1], height=means, align='edge', width=bin_size,
        edgecolor='white', color=facecolor_alpha)
    
    y = means
    x = (np.arange(1, y.size+1)) * 10 - 5
    params, covariance = curve_fit(f=f, xdata=x, ydata=y,
        p0=[0.185, 0.06, 0.02])
    y_fit = f(x, params[0], params[1], params[2])
    a0 = params[0] - params[1]
    a = np.quantile(y, 0.9)
    thr = a0 + (a-a0) * 0.75
    x_size = x[y_fit>thr][0]
    ax.axvline(x=x_size, color='black', linestyle='--', linewidth=1)
    ax.axhline(y=thr, color='black', linestyle='--', linewidth=1)

    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(labelsize=fontsize)
    ax.set_xlim(-bin_size*0.5, right_lim-bin_size*0.5)
    ax.set_ylim(bottom=0.1)
    plt.show()

def ori_dir(data):
    n = data.size
    angles = np.linspace(0, 2*np.pi, n+1)[:-1]
    
    ori_sum = np.sum(data*np.exp(2*1j*angles))
    pref_ori = np.mod(np.rad2deg(np.angle(ori_sum)/2),180)
    osi = np.abs(ori_sum/np.sum(data))
    dir_sum = np.sum(data*np.exp(1j*angles))
    pref_dir = np.mod(np.rad2deg(np.angle(dir_sum)),360)
    dsi = np.abs(dir_sum/np.sum(data))
    
    return pref_ori, osi, pref_dir, dsi

def cal_ori(data):
    n = data.size
    angles = np.linspace(0, np.pi, n+1)[:-1]
    
    ori_sum = np.sum(data*np.exp(2*1j*angles))
    pref_ori = np.mod(np.rad2deg(np.angle(ori_sum)/2),180)
    osi = np.abs(ori_sum/np.sum(data))

    return pref_ori, osi

''' test
data = np.random.rand(12)
data[4]=100
data[10]=100
plt.plot(data)
n = data.size
pref_ori, osi, pref_dir, dsi = ori_dir(data)
print([pref_ori, osi, pref_dir, dsi])
'''

def arrow_plot_bicolor(mb,rois_pos_all,roi_sel,type,only_pref=True,arrow_length_input=12, line_length_input=12, si_thr=0.1, title=None):
        
    figsize = (3, 3)
    fontsize = 20

    rois_pos = rois_pos_all[roi_sel,:]
    n_rois = rois_pos.shape[0]
    pref_dir = mb['pref_dir'][roi_sel]
    # pref_ori = mb['pref_ori'][roi_sel]
    dsi = mb['dsi'][roi_sel]
    # osi = mb['osi'][roi_sel]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-35, 530)
    ax.set_ylim(-35, 530)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for r in range(n_rois):
        if only_pref:
            arrow_length = arrow_length_input
            if dsi[r]>=si_thr:
                xy = (rois_pos[r][0] + math.cos(math.radians(pref_dir[r]))*arrow_length,
                    rois_pos[r][1] + math.sin(math.radians(pref_dir[r]))*arrow_length)
                xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_dir[r]))*arrow_length,
                        rois_pos[r][1] - math.sin(math.radians(pref_dir[r]))*arrow_length)
                ax.annotate('', xy=xy, xytext=xytext,
                            arrowprops=dict(color='blue', arrowstyle='->'))
            else:
                ax.scatter(rois_pos[r,0],rois_pos[r,1],s=5,marker='.',c='black',alpha=0.8)
        else:
            arrow_length = dsi[r]*100/12*line_length_input + 3
            if dsi[r]>=0.05:
                xy = (rois_pos[r][0] + math.cos(math.radians(pref_dir[r]))*arrow_length,
                    rois_pos[r][1] + math.sin(math.radians(pref_dir[r]))*arrow_length)
                xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_dir[r]))*arrow_length,
                        rois_pos[r][1] - math.sin(math.radians(pref_dir[r]))*arrow_length)
                if type[r]:
                    # arrowstyle='->, head_width=0.1',  linewidth=1,
                    ax.annotate('', xy=xy, xytext=xytext,
                                arrowprops=dict(color='red', arrowstyle='->, head_width={}, head_length={}'.format(dsi[r], dsi[r]*2), mutation_scale=10))
                else:
                    ax.annotate('', xy=xy, xytext=xytext,
                            arrowprops=dict(color='blue', arrowstyle='->, head_width={}, head_length={}'.format(dsi[r], dsi[r]*2), mutation_scale=10))
        
    ax.invert_yaxis()
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.suptitle(title)
    plt.show()

def cal_snr(x):
    '''
    x is (n_samples, n_repetition)
    '''
    snr = np.var(np.mean(x,axis=1))/np.mean(np.var(x,axis=1))
    return snr

def save_fig(file_path):
    file_ls = glob.glob(file_path + '*.*')
    if len(file_ls) >= 1:
        print('The images exist, and have been overwritten.')
    plt.savefig(file_path+'.png',bbox_inches='tight')
    plt.savefig(file_path+'.svg',bbox_inches='tight')
    plt.savefig(file_path+'.pdf',bbox_inches='tight')
    
def get_stim_time(file, threshold=0.8, plot=True, repeat=5):
    '''
    file: full path of the voltage recording file
    threshold: the detection highly depend on the threshold, 
    plot: plot the trigger signal for 
    '''

    # Read a voltage recording .csv file of trigger signal
    trigger_signal = pd.read_csv(file)
    # trigger_signal.info()
    # trigger_signal['Time(ms)']
    # trigger_signal[' Input 1'] # Note the 'space' before the 2nd column name.
    trigger_signal_max = trigger_signal[' Input 1'].max()
    threshold = trigger_signal_max * threshold

    if plot:
        time_s = trigger_signal['Time(ms)'] * 0.001 # time: ms to s
        xmin = 0
        xmax = time_s.iloc[-1]
        fig, axs = plt.subplots(2, 1)
        fig.set_size_inches(10, 3)
        axs[0].plot(time_s, trigger_signal[' Input 1'], label="voltage")
        axs[0].set_xlim(0, time_s.iloc[-1])
        axs[0].hlines(y=threshold, xmin=xmin, xmax=xmax, color='r', label="threshold")
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('ISI of triggers (s)')
        axs[0].legend(loc='upper right')

        xmax = 10 # time in second
        axs[1].plot(time_s, trigger_signal[' Input 1'], label="voltage")
        axs[1].set_xlim(xmin, xmax)
        axs[1].hlines(y=threshold, xmin=0, xmax=xmax, color='r', label="threshold")
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('ISI of triggers (s)')
        axs[1].legend(loc='upper right')
        # trigger_signal.plot(x='Time(ms)', y=' Input 1', xlabel='Time (s)', ylabel='Voltage (mV)', figsize=(20, 5))
        # trigger_signal.plot(x='Time(ms)', y=' Input 1', xlabel='Time (s)', ylabel='Voltage (mV)', xlim=(0, 10000), figsize=(20, 5))

    # pandas.DataFrame.gt, Get Greater than of dataframe and other.
    trigger_signal['above'] = trigger_signal[' Input 1'].gt(threshold).astype(int)

    # Detect the trigger time
    trigger_type = 1 # 1, detect the  rising edge; -1, detect the falling edge
    stim_time_index = trigger_signal.index[trigger_signal['above'].diff() == trigger_type].tolist()
    stim_time_ms = trigger_signal['Time(ms)'][stim_time_index] # index to stimulus time, in ms
    stim_time_s = pd.Series(stim_time_ms * 0.001, name='Time(s)') # stimulus time in s

    # if (stim_time_s.size % repeat): # if the size of stim_time_s not 
    #     print('NOTE: There may be some mistake...')

    if plot:            
        # plot the ISI of stimulus time
        ISI = stim_time_s.diff()
        # ISI = ISI[ISI > 0.1] # delete the false trigger with ISI <=0.1
        fig, ax = plt.subplots()
        ax.plot(range(ISI.size), ISI, '.')
        # ISI.plot(x=range(ISI.size), y='Time(s)', xlabel='Triggers', ylabel='ISI of triggers (s)', figsize=(20, 5))
        # plt.plot(range(ISI.size), ISI, '.')
        ax.set_xlabel('Triggers')
        ax.set_ylabel('ISI of triggers (s)')
        fig.set_size_inches(10, 3)

    return stim_time_s.to_numpy()

# def sorting(data, trigger, seq):
#     '''
#     data: ndarray,
#     trigger: the index of each trigger
#     seq: ndarray
#     '''
#     [n_roi, n_sample] = data.shape
#     n_seq = seq.size
#     n_stim = np.unique(seq).size
#     n_rep = int(n_seq/n_stim)
#     n_trigger = trigger.size
#     if n_trigger != n_stim * n_rep:
#         print('n_trigger != n_stim * n_rep')
#     stim_len = round(np.median(trigger[1:] - trigger[:-1]) + 1) # The reason for +1 here is to get enough trace length
#     data_sorted = np.zeros((stim_len, n_stim, n_rep+1, n_roi+1))
#     for i_roi in range(n_roi):
#         for i_trigger in range(n_trigger):
#             i_rep = int(i_trigger/n_stim)
#             data_sorted[:,seq[i_trigger],i_rep,i_roi] = data[i_roi,trigger[i_trigger]:trigger[i_trigger]+stim_len]
#             # print(n_trigger)
    
#     data_sorted[:,:,n_rep,:] = np.mean(data_sorted[:,:,:n_rep, :],axis=2)
#     data_sorted[:,:,:,n_roi] = np.mean(data_sorted[:,:,:,:n_roi],axis=3)
    
#     return data_sorted

def sorting(data, trigger, seq, pre_onset=0):
    '''
    data: ndarray,
    trigger: the index of each trigger
    seq: ndarray
    pre_onset: int, the number of frames before onset
    '''
    [n_roi, n_sample] = data.shape
    n_seq = seq.size
    n_stim = np.unique(seq).size
    n_rep = int(n_seq/n_stim)
    n_trigger = trigger.size
    if n_trigger != n_stim * n_rep:
        print('n_trigger != n_stim * n_rep')
    stim_len = round(np.median(trigger[1:] - trigger[:-1]) + 1) # The reason for +1 here is to get enough trace length
    stim_len = stim_len + pre_onset
    trigger = trigger - pre_onset
    data_sorted = np.zeros((stim_len, n_stim, n_rep+1, n_roi+1))
    for i_roi in range(n_roi):
        for i_trigger in range(n_trigger):
            i_rep = int(i_trigger/n_stim)
            data_sorted[:,seq[i_trigger],i_rep,i_roi] = data[i_roi,trigger[i_trigger]:trigger[i_trigger]+stim_len]
            # print(n_trigger)
    
    data_sorted[:,:,n_rep,:] = np.mean(data_sorted[:,:,:n_rep, :],axis=2)
    data_sorted[:,:,:,n_roi] = np.mean(data_sorted[:,:,:,:n_roi],axis=3)
    
    return data_sorted

def find_peak(data, win_index=None, baseline=0, signal_deflect=1):
    '''
    find peak at axis 0
    '''
    if win_index == None:
        win_index = [0,data.shape[0]]
    if signal_deflect == 1:
        peak = np.max(data[win_index[0]:win_index[1]],axis=0) - baseline
    else:
        peak = baseline - np.min(data[win_index[0]:win_index[1]],axis=0)
    peak = peak * (peak > 0)
    return peak

# def mean_resp(data, win_index=None, baseline=0, signal_deflect=1):
#     '''
#     measure the mean at axis 0
#     '''
#     if win_index == None:
#         win_index = [0,data.shape[0]]
#     if signal_deflect == 1:
#         ave = np.mean(data[win_index[0]:win_index[1]],axis=0) - baseline
#     else:
#         ave = baseline - np.mean(data[win_index[0]:win_index[1]],axis=0)
#     ave = ave * (ave > 0)
#     return ave

def mean_resp(data, win_index=None, baseline=0, signal_deflect=1):
    '''
    measure the mean at axis 0
    '''
    print(data.shape)
    if win_index == None:
        win_index = [0, data.shape[0]]
    if signal_deflect == 1:
        above_baseline = (data[win_index[0]:win_index[1]] > baseline)
        # print(above_baseline.shape, data[win_index[0]:win_index[1]].shape)
        data_above_baseline = data[win_index[0]:win_index[1]] * above_baseline
        # print(data_above_baseline.shape)
        # ave = np.sum(data_above_baseline, axis=0) / np.sum(above_baseline, axis=0)
        ave = np.mean(data_above_baseline, axis=0)
        # print(ave.shape)
    else:
        below_baseline = (data[win_index[0]:win_index[1]] < baseline)
        data_below_baseline = data[win_index[0]:win_index[1]] * below_baseline
        ave = np.sum(data_below_baseline, axis=0) / np.sum(below_baseline, axis=0)
        ave = - ave
    # ave = ave * (ave > 0)
    return ave

def trace_compare(data, scale_bar=False):
    '''
    plot the traces of before and after
    data: 2 x n x m, 2 for before and after, n for time points, m for repeats
    '''
    fig, ax = plt.subplots(figsize=(5, 3), layout='constrained')

    mean = np.mean(data, axis=-1)

    y1 = mean + np.std(data, axis=-1)
    y2 = mean - np.std(data, axis=-1)

    x = np.arange(data.shape[1])

    ax.plot(x, mean[0], color='black', lw=1)
    ax.plot(x, mean[1], color='blue', lw=1)

    ax.fill_between(x, y1[0], y2[0], alpha=0.3, color='black')
    ax.fill_between(x, y1[1], y2[1], alpha=0.3, color='blue')

    ax.hlines(y=0, color='red', lw=0.5, linestyle='--', xmin=0, xmax=len(x)-1)

    if scale_bar:
        scale_x = 0.5 # in seconds
        scale_w = scale_x*10
        start_x = len(x)

        scale_y = 0.6
        y_range = np.max(y1) - np.min(y2)
        scale_h = scale_y/y_range

        ax.vlines(x=start_x, ymin=0, ymax=scale_h, color='black', lw=1)
        ax.hlines(y=0, xmin=start_x, xmax=start_x+scale_w, color='black', lw=1)

    ax.set_axis_off()
    plt.show()

def twoD_gaussion_func (xy,x0,y0,theta,sigma_x,sigma_y,amp,base):
    x0 = float(x0)
    y0 = float(y0)
    a = np.cos(theta)**2 / (2 * sigma_x**2) + np.sin(theta)**2 / (2 * sigma_y**2)
    b = np.sin(2 * theta) / (4 * sigma_y**2) - np.sin(2 * theta) / (4 * sigma_x**2)
    c = np.sin(theta)**2 / (2 * sigma_x**2) + np.cos(theta)**2 / (2 * sigma_y**2)
    z = amp * np.exp(-(a * ((xy[0] - x0)**2) + 2 * b * (xy[0] - x0) * (xy[1] - y0) + c * ((xy[1] - y0)**2)))
    return z


def GridInterpolator(rf_2d, interp_num=5,method='linear'):
    n_row,n_col = rf_2d.shape
    x = np.linspace(1,n_col,n_col)
    y = np.linspace(1,n_row,n_row)

    x_2d, y_2d = np.meshgrid(x,y,indexing='xy')
    
    # interp = RegularGridInterpolator((x,y),rf_2d,bounds_error=False, fill_value=None,method=method)  

    xx = np.linspace(0.5+0.5/interp_num,n_col+0.5-0.5/interp_num,n_col*interp_num)
    yy = np.linspace(0.5+0.5/interp_num,n_row+0.5-0.5/interp_num,n_row*interp_num)
    xx_2d, yy_2d = np.meshgrid(xx,yy,indexing='xy')
    # rf_2d_interp = interp((xx_2d,yy_2d))
    rf_2d_interp = scipy.ndimage.zoom(rf_2d,interp_num,order=3)
    return xx_2d,yy_2d,rf_2d_interp
# fig,ax=plt.subplots()
# ax.pcolormesh(xx_2d,yy_2d,rf_2d_interp)
# ax.set_aspect('equal')
# ax.invert_yaxis()
# plt.show()

# def rf_gaussian_fit(xy_2d,rf_2d,x0=3,y0=2,theta=0,sigma_x=1,sigma_y=1,amp=1,base=0):
def rf_gaussian_fit(xy_2d,rf_2d,x0,y0,theta,sigma_x,sigma_y,amp,base):
    n_row,n_col = rf_2d.shape
    # x = np.linspace(1,n_row,n_row)
    # y = np.linspace(1,n_col,n_col)
    # x_2d, y_2d = np.meshgrid(x,y,indexing='ij')
    x_2d,y_2d = xy_2d
    x_1d = x_2d.ravel()
    y_1d = y_2d.ravel()
    xy_1d = np.vstack((x_1d, y_1d))
    rf_1d = rf_2d.ravel()
    # popt, pcov = curve_fit(twoD_gaussion_func, xy_1d, rf_1d, maxfev=int(1e5), bounds = ([0,0,-np.inf,-np.inf,-np.inf,-np.inf,n-p.inf], [4,6,2,2,2,1,1]))
    popt, pcov = curve_fit(twoD_gaussion_func, xy_1d, rf_1d, maxfev=int(1e5), bounds = ([0, 0, -1000, 0, 0, -5, 0], [6, 4, 1000, 2, 2, 5, 2])) #[x0,y0,theta,sigma_x,sigma_y,amp,base]
    # popt, pcov = curve_fit(twoD_gaussion_func, xy_1d, rf_1d, maxfev=int(1e5))
    # print(popt)
    rf_1d_fit = twoD_gaussion_func(xy_1d, *popt)
    residuals = rf_1d - rf_1d_fit
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((rf_1d-np.mean(rf_1d))**2)
    r_squared = 1 - (ss_res / ss_tot)
    rf_2d_fit = rf_1d_fit.reshape((n_row,n_col))
    x0, y0, theta, sigma_x, sigma_y, amp, base = popt
    area = np.pi*2*np.log(10)*np.abs(sigma_x*sigma_y)
    rf_fit_para = {
        'x0': x0,
        'y0': y0,
        'theta': theta/np.pi*180,
        'sigma_x': np.abs(sigma_x),
        'sigma_y': np.abs(sigma_y),
        'amp': amp,
        'base': base,
        'area': area,
        'r2':r_squared}
    return rf_fit_para, rf_2d_fit

def rf_process(rf_2d,interp_num=5,visual_center=(95,25),unit_size=10,r2_thr=0.5,plot=False):
    rf_2d = rf_2d - np.mean(rf_2d)
    rf_2d = np.where (rf_2d < (np.max(rf_2d)/3), 0, rf_2d)
    n_row,n_col = rf_2d.shape
    xx_2d,yy_2d,rf_2d_interp = GridInterpolator(rf_2d, interp_num=interp_num)
    n_row_interp,n_col_interp = rf_2d_interp.shape
    if plot:
        fig,ax=plt.subplots()
        ax.pcolormesh(xx_2d,yy_2d,rf_2d_interp)
        ax.set_aspect('equal')
        ax.invert_yaxis()
        plt.show()
    i_row = np.argmax(np.max(rf_2d_interp, axis=0))
    i_col = np.argmax(np.max(rf_2d_interp, axis=1))
    x0 = xx_2d[i_col,i_row]
    y0 = yy_2d[i_col,i_row]
    # print((x0,y0))
    theta = 0
    sigma_x = 1
    sigma_y = 1
    amp = np.max(rf_2d)
    base = 0
    try:
        rf_fit_para, rf_2d_fit = rf_gaussian_fit((xx_2d,yy_2d), rf_2d_interp, x0=x0,y0=y0, theta=theta, sigma_x=sigma_x,sigma_y=sigma_y,amp=amp,base=base)
        if plot:
            fig,ax=plt.subplots()
            ax.pcolormesh(xx_2d,yy_2d,rf_2d_fit)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            plt.show()
        rf_fit_para['x0'] = (rf_fit_para['x0']-(n_col+1)/2)*unit_size+visual_center[0]
        rf_fit_para['y0'] = -(rf_fit_para['y0']-(n_row+1)/2)*unit_size+visual_center[1]
        rf_fit_para['sigma_x'] = rf_fit_para['sigma_x']*unit_size
        rf_fit_para['sigma_y'] = rf_fit_para['sigma_y']*unit_size
        rf_fit_para['area'] = rf_fit_para['area']*unit_size**2
        # print(rf_fit_para)
        # print(rf_2d_fit)
    except :
        rf_fit_para = {'x0':np.nan, 'y0':np.nan, 'theta':0, 'sigma_x': 0, 'sigma_y': 0, 'amp': 0, 'base': 0, 'area': 0, 'r2':0}
        rf_2d_fit = np.zeros((n_row_interp,n_col_interp))
        print('wrong') 
    if rf_fit_para['r2']<r2_thr:
        rf_fit_para['x0']=np.nan
        rf_fit_para['y0']=np.nan
        rf_fit_para['area']=0
    # rf_fit_para, rf_2d_fit = rf_gaussian_fit((xx_2d,yy_2d),rf_2d_interp,x0=x0,y0=y0,theta=theta,sigma_x=sigma_x,sigma_y=sigma_y,amp=amp,base=base)
    # plt.imshow(rf_2d_fit)
    # plt.show()
    # rf_fit_para['x0'] = (rf_fit_para['x0']-(n_col+1)/2)*unit_size+visual_center[0]
    # rf_fit_para['y0'] = (rf_fit_para['y0']-(n_row+1)/2)*unit_size+visual_center[1]
    # rf_fit_para['sigma_x'] = rf_fit_para['sigma_x']*unit_size
    # rf_fit_para['sigma_y'] = rf_fit_para['sigma_y']*unit_size
    # rf_fit_para['area'] = rf_fit_para['area']*unit_size**2
    # # print(rf_fit_para)
    # # print(rf_2d_fit)
    
    return rf_fit_para, rf_2d_fit

def arrow_plot(mb,rois_pos_all,roi_sel,only_pref=True,arrow_length_input=12, si_thr=0.1, title=None):
    rois_pos = rois_pos_all[roi_sel,:]
    n_rois = rois_pos.shape[0]
    pref_dir = mb['pref_dir'][roi_sel]
    # pref_ori = mb['pref_ori'][roi_sel]
    dsi = mb['dsi'][roi_sel]
    # osi = mb['osi'][roi_sel]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-10,512)
    ax.set_ylim(-10,512)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for r in range(n_rois):
        if only_pref:
            arrow_length = arrow_length_input
            if dsi[r]>=si_thr:
                xy = (rois_pos[r][0] + math.cos(math.radians(pref_dir[r]))*arrow_length,
                    rois_pos[r][1] + math.sin(math.radians(pref_dir[r]))*arrow_length)
                xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_dir[r]))*arrow_length,
                        rois_pos[r][1] - math.sin(math.radians(pref_dir[r]))*arrow_length)
                ax.annotate('', xy=xy, xytext=xytext,
                            arrowprops=dict(color='blue', arrowstyle='->'))
            else:
                ax.scatter(rois_pos[r,0],rois_pos[r,1],s=5,marker='.',c='black',alpha=0.8)
        else:
            arrow_length = dsi[r]*100/12*arrow_length_input + 3
            xy = (rois_pos[r][0] + math.cos(math.radians(pref_dir[r]))*arrow_length,
                rois_pos[r][1] + math.sin(math.radians(pref_dir[r]))*arrow_length)
            xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_dir[r]))*arrow_length,
                    rois_pos[r][1] - math.sin(math.radians(pref_dir[r]))*arrow_length)
            ax.annotate('', xy=xy, xytext=xytext,
                        arrowprops=dict(color='blue', arrowstyle='->'))
        
    fig.set_size_inches(8, 8)
    ax.invert_yaxis()
    plt.suptitle(title)
    plt.show()

def arrow_line_plot(mb,rois_pos_all,roi_sel,only_pref=True,arrow_length_input=12, line_length_input=12, si_thr = 0.1, title=None):
    rois_pos = rois_pos_all[roi_sel,:]
    n_rois = rois_pos.shape[0]
    pref_dir = mb['pref_dir'][roi_sel]
    pref_ori = mb['pref_ori'][roi_sel]
    dsi = mb['dsi'][roi_sel]
    osi = mb['osi'][roi_sel]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-10,512)
    ax.set_ylim(-10,512)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for r in range(n_rois):
        if only_pref:
            arrow_length = arrow_length_input
            if dsi[r]>=si_thr:
                xy = (rois_pos[r][0] + math.cos(math.radians(pref_dir[r]))*arrow_length,
                    rois_pos[r][1] + math.sin(math.radians(pref_dir[r]))*arrow_length)
                xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_dir[r]))*arrow_length,
                        rois_pos[r][1] - math.sin(math.radians(pref_dir[r]))*arrow_length)
                ax.annotate('', xy=xy, xytext=xytext,
                            arrowprops=dict(color='blue', arrowstyle='->'))
            else:
                ax.scatter(rois_pos[r,0],rois_pos[r,1],s=5,marker='.',c='black',alpha=0.8)
        else:
            arrow_length = dsi[r]*100/12*arrow_length_input
            xy = (rois_pos[r][0] + math.cos(math.radians(pref_dir[r]))*arrow_length,
                rois_pos[r][1] + math.sin(math.radians(pref_dir[r]))*arrow_length)
            xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_dir[r]))*arrow_length,
                    rois_pos[r][1] - math.sin(math.radians(pref_dir[r]))*arrow_length)
            ax.annotate('', xy=xy, xytext=xytext,
                        arrowprops=dict(color='blue', arrowstyle='->'))
        
        if only_pref:
            line_length = line_length_input
            if osi[r]>=si_thr:
                xy = (rois_pos[r][0] + math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                    rois_pos[r][1] + math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
                xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                        rois_pos[r][1] - math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
                ax.annotate('', xy=xy, xytext=xytext,
                            arrowprops=dict(color='red', arrowstyle='-'))
            else:
                ax.scatter(rois_pos[r,0],rois_pos[r,1],s=5,marker='.',c='green',alpha=0.8)
        else:
            line_length = osi[r]*100/12*line_length_input
            xy = (rois_pos[r][0] + math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                rois_pos[r][1] + math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
            xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                    rois_pos[r][1] - math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
            ax.annotate('', xy=xy, xytext=xytext,
                        arrowprops=dict(color='red', arrowstyle='-'))
        
    fig.set_size_inches(8, 8)
    ax.invert_yaxis()
    plt.suptitle(title)
    plt.show()
    
def line_plot(mb,rois_pos_all,roi_sel,only_pref=True,line_length_input=12, si_thr=0.1, title=None):
    rois_pos = rois_pos_all[roi_sel,:]
    n_rois = rois_pos.shape[0]
    pref_ori = mb['pref_ori'][roi_sel]
    osi = mb['osi'][roi_sel]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-10,512)
    ax.set_ylim(-10,512)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for r in range(n_rois):
        if only_pref:
            line_length = line_length_input
            if osi[r]>=si_thr:
                xy = (rois_pos[r][0] + math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                    rois_pos[r][1] + math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
                xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                        rois_pos[r][1] - math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
                ax.annotate('', xy=xy, xytext=xytext,
                            arrowprops=dict(color='red', arrowstyle='-'))
            else:
                ax.scatter(rois_pos[r,0],rois_pos[r,1],s=5,marker='.',c='green',alpha=0.8)
        else:
            line_length = osi[r]*100/12*line_length_input
            xy = (rois_pos[r][0] + math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                rois_pos[r][1] + math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
            xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                    rois_pos[r][1] - math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
            ax.annotate('', xy=xy, xytext=xytext,
                        arrowprops=dict(color='red', arrowstyle='-'))
        
    fig.set_size_inches(8, 8)
    ax.invert_yaxis()
    plt.suptitle(title)
    plt.show()
# test the function
# pref_dir = np.arange(0,12)*30
# rois_pos = np.stack((np.arange(0,11),np.arange(0,11)),axis=1)*40
# arrow_plot(pref_dir,pref_dir,rois_pos,arrow_length_input=np.arange(1,12)/12,line_length_input=np.arange(1,12)/12)    
    
def line_plot_bicolor(data_dict,rois_pos_all,roi_sel,cell_type,only_pref=True,line_length_input=12, si_thr=0.1, title=None):
    '''
    dict: dictionary, {'pref_ori': np.array, 'osi': np.array}
    only_pref: True, the line length is equal; False, the line length show OSI
    cell_type: cell type, if true, plot line as red, if false, plot lien as blue
    '''
    figsize = (3, 3)
    fontsize = 20
    rois_pos = rois_pos_all[roi_sel,:]
    n_rois = rois_pos.shape[0]
    pref_ori = data_dict['pref_ori'][roi_sel]
    osi = data_dict['osi'][roi_sel]
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-35, 530)
    ax.set_ylim(-35, 530)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for r in range(n_rois):
        if only_pref:
            line_length = line_length_input
            if osi[r]>=si_thr:
                xy = (rois_pos[r][0] + math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                    rois_pos[r][1] + math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
                xytext =  (rois_pos[r][0] - math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                        rois_pos[r][1] - math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
                ax.annotate('', xy=xy, xytext=xytext,
                            arrowprops=dict(color='red', arrowstyle='-'))
            else:
                ax.scatter(rois_pos[r,0],rois_pos[r,1],s=5,marker='.',c='green',alpha=0.8)
        else:
            line_length = osi[r]*100/12*line_length_input
            xy = (rois_pos[r][0] + math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                rois_pos[r][1] + math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
            xytext = (rois_pos[r][0] - math.cos(math.radians(pref_ori[r])-np.pi/2)*line_length,
                    rois_pos[r][1] - math.sin(math.radians(pref_ori[r])-np.pi/2)*line_length)
            if cell_type[r]:
                
                ax.annotate('', xy=xy, xytext=xytext,
                            arrowprops=dict(color='red', arrowstyle='-'))
            else:
                ax.annotate('', xy=xy, xytext=xytext,
                            arrowprops=dict(color='blue', arrowstyle='-'))
        
    ax.invert_yaxis()
    ax.tick_params(axis='both', labelsize=fontsize)
    plt.suptitle(title)
    plt.show()

def amp_plot(amp,rois_pos,figsize=(5,5),x_lim=(0,512),y_lim=(0,512),alpha=1,cmap='viridis',yaxis_invert=True,title=None):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # n_rois = len(rois)
    fig = plt.figure(figsize=figsize)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    s = ax.scatter(rois_pos[:,0],rois_pos[:,1],s=10, c=amp,cmap=cmap,alpha=alpha)
    if yaxis_invert:
        ax.invert_yaxis()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="5%")
    plt.colorbar(s, cax=cax)
    if label:
        ax.set_xlabel('AM-PL (µm)')
        ax.set_ylabel('AL-PM (µm)')
        ax.set_title(title)
        cax.set_ylabel('Visual position (Deg)')
    plt.tight_layout()
    plt.show()

def amp_plot_v2(amp, rois_pos, figsize=(5,5), x_lim=(0,512), y_lim=(0,512),
    alpha=1, cmap='viridis', yaxis_invert=True, title=None, label=True):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    # n_rois = len(rois)

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    s = ax.scatter(rois_pos[:,0],rois_pos[:,1],s=10, c=amp,cmap=cmap,alpha=alpha)
    if yaxis_invert:
        ax.invert_yaxis()

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad="5%")
    plt.colorbar(s, cax=cax)
    if label:
        # ax.set_xlabel('AM-PL (µm)')
        # ax.set_ylabel('AL-PM (µm)')
        ax.set_title(title)
        cax.set_ylabel('Visual position (Deg)')
    plt.tight_layout()
    plt.show()

def rois_plot(rois_pos_list,figsize=(8,8),x_lim=(0,512),y_lim=(0,512),c='blue',alpha=1,title=None,legend=['0','90','180','270']):
    # n_rois = len(rois)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i,rois_pos in enumerate(rois_pos_list):
        if rois_pos.shape[0]==2:
            ax.scatter(rois_pos[0,:],rois_pos[1,:],c=c[i],alpha=alpha)
        else:
            ax.scatter(rois_pos[:,0],rois_pos[:,1],c=c[i],alpha=alpha)
        
    ax.legend(legend)
    ax.invert_yaxis()  
    plt.suptitle(title)
    plt.show()

def amp_plot_bicolor(amp, rois_pos, is_exc, is_inh, figsize=(5, 5), x_lim=(-10,530), y_lim=(-10,530), alpha=1, exclude_extremum=True, yaxis_invert=True, vmax=None, title=None):
    good = np.logical_not(np.isnan(amp)) # not nan
    if exclude_extremum is not None:
        not_one = abs(amp) < 0.95 # # to exclude the values around 1
        good = np.logical_and(good, not_one)
    is_exc = np.logical_and(good, is_exc)
    is_inh = np.logical_and(good, is_inh)
    fontsize = 12
    fig = plt.figure(figsize=figsize) # 
    subfigs = fig.subfigures(1, 2, wspace=0.01, width_ratios=[30, 1])

    ax_scatter = subfigs[0].subplots()
    ax_scatter.set_aspect('equal')
    ax_scatter.set_xlim(x_lim)
    ax_scatter.set_ylim(y_lim)
    ax_scatter.spines['top'].set_visible(False)
    ax_scatter.spines['right'].set_visible(False)
    
    vmax = np.max((np.max(amp[is_exc]), np.max(amp[is_inh])))

    # for inhibitory rois
    # print(np.max(amp[is_inh]), np.min(amp[is_inh]))
    cmap = mcolors.LinearSegmentedColormap.from_list('', ['white','blue'])
    _amp = amp[is_inh]
    _roi_pos = rois_pos[is_inh, :]
    idx = np.argsort(_amp)

    s_inh = ax_scatter.scatter(_roi_pos[idx, 0],_roi_pos[idx, 1],c=_amp[idx],cmap=cmap,alpha=alpha, edgecolors='blue', linewidths=0.3, vmax=vmax) # edgecolors='lightgray', linewidths=0.3

    # for excitatory rois
    # print(np.max(amp[is_exc]), np.min(amp[is_exc]))
    cmap = mcolors.LinearSegmentedColormap.from_list('', ['white', 'red'])
    _amp = amp[is_exc]
    _roi_pos = rois_pos[is_exc, :]
    idx = np.argsort(_amp)

    s_exc = ax_scatter.scatter(_roi_pos[idx, 0],_roi_pos[idx, 1],c=_amp[idx],cmap=cmap,alpha=alpha, edgecolors='red', linewidths=0.3, vmin=-0.45) # edgecolors='lightgray', linewidths=0.3
    
    if yaxis_invert:
        ax_scatter.invert_yaxis()
    # fig.colorbar(s)
    ax_scatter.tick_params(axis='both', labelsize=fontsize)

    axis_colorbar = subfigs[1].subplots(2, 1)
    fig.colorbar(s_exc, cax=axis_colorbar[0], location='left', orientation='vertical')
    axis_colorbar[0].tick_params(axis='both', labelsize=10)
    locs = [-0.3, 0, 0.3]
    # axis_colorbar[0].set_yticks(locs)
    # axis_colorbar[0].set_ylabel('Excitatory')
    fig.colorbar(s_inh, cax=axis_colorbar[1], location='left', orientation='vertical')
    axis_colorbar[1].tick_params(axis='both', labelsize=10)
    # axis_colorbar[1].set_ylabel('Inhibitory')
    # axis_colorbar[1].set_yticks(locs)
    plt.suptitle(title)
    # plt.tight_layout()
    plt.show()
    
def rf_pos_plot(rois_pos_list,figsize=(8,8),x_lim=(50,140),y_lim=(-20,70),c='blue',alpha=1,title=None,legend=['0','90','180','270']):
    # n_rois = len(rois)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for i,rois_pos in enumerate(rois_pos_list):
        if rois_pos.shape[0]==2:
            ax.scatter(rois_pos[0,:],rois_pos[1,:],c=c[i],alpha=alpha)
        else:
            ax.scatter(rois_pos[:,0],rois_pos[:,1],c=c[i],alpha=alpha)
    ax.legend(legend)
    plt.suptitle(title)
    plt.show()
    return fig

def plot_response_scalebar(data, window=[0, 10], scale_bar=False, title=None,
    indicate=[], auto_baseline=False):
    '''
    data shape in (n_cols, stim_len, cell, repetition)
    indicate_time: the time point to indicate the stimulus
    '''
    # indicate = [0, 10, 20, 30, 40]
    xlim_min = 0
    if len(data.shape) == 3:
        n_cols = 1
        xlim_max = data.shape[0]
        n_rows = data.shape[1]
    if len(data.shape) == 4:
        n_cols = data.shape[0]
        xlim_max = data.shape[1]
        n_rows = data.shape[2]
    # ylim_min = np.min(data)
    # ylim_max = np.max(data)
    # print('ylim_max - ylim_min = {}'.format(ylim_max-ylim_min))
    if n_cols == 1:
        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, layout='constrained', figsize=(n_cols*2, n_rows))

        ylim_min = 0
        ylim_max = 0
        for r in range(n_rows):

            _temp = data[:, r, :]
            mean = np.mean(_temp, axis=1)
            if auto_baseline:
                mean_baseline = np.mean(mean[:5])
                mean = mean - mean_baseline
            y1 = mean + np.std(_temp, axis=1)
            y2 = mean - np.std(_temp, axis=1)
            axis[r].fill_between(np.arange(xlim_max), y1, y2, color='black', alpha=0.3)
            axis[r].plot(mean, color='black')
            axis[r].set_xlim(left=xlim_min)
            axis[r].set_xticks([])
            axis[r].set_yticks([])
            axis[r].spines['top'].set_visible(False)
            axis[r].spines['right'].set_visible(False)
            axis[r].spines['left'].set_visible(False)
            axis[r].spines['bottom'].set_visible(False)
            axis[r].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')

            y = axis[r].get_ylim()
            ylim_min = np.min((y[0], ylim_min))
            ylim_max = np.max((y[1], ylim_max))

        # Unify the y-axis limit of the different rows
        for r in range(n_rows):
            axis[r].set_ylim(ylim_min, ylim_max)

        # add bars at the right side
        if scale_bar:
            top = 0.99
            right = 0.99
            hline_length = 0.2
            vline_length = 0.3 / (ylim_max - ylim_min)
            axis[r].axhline(y = ylim_min + (ylim_max-ylim_min)*top, xmin=right-hline_length, xmax=right, color='black', linewidth=1)
            # axis[i].text(0.15, ylim_min - 0.1, '10 units', fontsize=12)
            axis[r].axvline(x = right*xlim_max, ymin=top-vline_length, ymax=top, color='black', linewidth=1)
            # axis[i].text(xlim_min - 0.5, 0.15, '0 ms', fontsize=12)

    if n_cols > 1:
        fig, axis = plt.subplots(nrows=n_rows, ncols=n_cols, layout='constrained', figsize=(n_cols*2, n_rows))
        for r in range(n_rows):
            for c in range(n_cols):
                _temp = data[c, :, r, :]
                mean = np.mean(_temp, axis=1)
                if auto_baseline:
                    mean_baseline = np.mean(mean[:5])
                    mean = mean - mean_baseline
                y1 = mean + np.std(_temp, axis=1)
                y2 = mean - np.std(_temp, axis=1)
                axis[r, c].fill_between(np.arange(xlim_max), y1, y2, color='black', alpha=0.3)
                axis[r, c].plot(mean, color='black')
                # axis[r, c].set_xlim(left=xlim_min)
                # axis[r, c].set_ylim(ylim_min, ylim_max)
                axis[r, c].set_xticks([])
                axis[r, c].set_yticks([])
                axis[r, c].spines['top'].set_visible(False)
                axis[r, c].spines['right'].set_visible(False)
                axis[r, c].spines['left'].set_visible(False)
                axis[r, c].spines['bottom'].set_visible(False)
                axis[r, c].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')
                # axis[r, c].set_ylabel('{}'.format(i+1))
                for ind in indicate:
                    axis[r, c].axvline(x=ind, color='black', linewidth=1, linestyle='--', alpha=0.5)

            # Unify the y-axis limit of each row
            ylim_min = 0
            ylim_max = 0
            for c in range(n_cols):
                y = axis[r, c].get_ylim()
                ylim_min = np.min((y[0], ylim_min))
                ylim_max = np.max((y[1], ylim_max))
            for c in range(n_cols):
                axis[r, c].set_ylim(ylim_min, ylim_max)

            # add bars at the last columns
            if scale_bar:
                left = xlim_max * 1.05
                x_scale = 0.5 # in seconds
                y_scale = 0.3
                scale_h = y_scale / (ylim_max - ylim_min)
                axis[r, c].hlines(y=0, xmin=left, xmax=left+10*x_scale,
                    color='black', linewidth=1)
                axis[r, c].vlines(x=left, ymin=-0.01, ymax=scale_h, color='black',
                    linewidth=1)

    plt.suptitle(title)

def plot_rfs(data,rois,n_cols=10,vmax=None,vmin=None, title=None): 
    n_roi = len(rois)
    [n_row_data,n_col_data] = data.shape[:2]
    row_col_ratio_data = n_row_data/n_col_data
    n_rows = math.ceil(n_roi/n_cols)
    # plot the RF
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={'xticks': [], 'yticks': []}, figsize=(n_cols*2, n_rows*row_col_ratio_data*1.2*2))
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    for row in range(n_rows):
        for col in range(n_cols):
            i = row*n_cols+col
            if i < n_roi:
                roi_index = rois[i]
                if n_rows>1:
                    if vmax is None or vmin is None:
                        axs[row][col].imshow(data[:, :, roi_index])
                    else:
                        axs[row][col].imshow(data[:, :, roi_index],vmin=vmin[roi_index],vmax=vmax[roi_index])
                    axs[row][col].set_xlabel('{}'.format(roi_index),labelpad=2)
                else:
                    if vmax is None or vmin is None:
                        axs[col].imshow(data[:, :, roi_index])
                    else:
                        axs[col].imshow(data[:, :, roi_index],vmin=vmin[roi_index],vmax=vmax[roi_index])
                    axs[col].set_xlabel('{}'.format(roi_index),labelpad=2)
    plt.suptitle(title,y=0.99)
    plt.show()
    
def plot_rfs_example(rf,rois, title=None):    
    n_rows = 1
    if type(rf)==tuple:
        n_cols = len(rf)
        figsize = (3*n_cols,3)
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={'xticks': [], 'yticks': []}, figsize=figsize)
        vmin = np.asarray(rf)[:,:,:,rois].min()
        vmax = np.asarray(rf)[:,:,:,rois].max()
        for i in range(n_cols):
            axs[i].imshow(rf[i][:, :, rois],vmin=vmin, vmax=vmax,cmap='viridis')
    else:
        n_cols = 1
        figsize = (4,3)
        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={'xticks': [], 'yticks': []}, figsize=figsize)
        axs.imshow(rf[:, :, rois],cmap='viridis')
    plt.suptitle(title)
    plt.show()
    
def plot_polar_graphs_example(data, rois, figsize=(3,3), color='black', title=None):
    n_rows = 1
    n_cols = 1
    angles = np.arange(0, 361, 30)
    theta = np.deg2rad(angles)
    # plot the polar graph
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={'rticks': [], 'xticks': [], 'projection':"polar"}, figsize=figsize)
    response = data[:, rois]
    response = np.append(response, response[0]) # append a element for plot a closed response
    axs.plot(theta, response,color=color)
    # axs.text(np.deg2rad(225),response.max()*1.5,'{}'.format(rois),fontsize=8)
    # axs.tick_params(labelbottom=False)
    axs.tick_params(labelleft=False)
    axs.set_theta_direction(-1)
    plt.suptitle(title)
                    
def plot_polar_graphs(data, rois, n_cols=10, title=None):
    n_roi = len(rois)
    n_rows = math.ceil(n_roi / n_cols)
    angles = np.arange(0, 361, 30)
    theta = np.deg2rad(angles)
    # plot the polar graph
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, subplot_kw={'rticks': [], 'xticks': [], 'projection':"polar"}, figsize=(n_cols, n_rows))
    for row in range(n_rows):
        for col in range(n_cols):
            i = row*n_cols+col
            if i < n_roi:
                roi_index = rois[i]
                response = data[:, roi_index]
                response = np.append(response, response[0]) # append a element for plot a closed response
                if n_rows>1:
                    axs[row][col].plot(theta, response)
                    axs[row][col].text(np.deg2rad(225),response.max()*1.5,'{}'.format(roi_index),fontsize=8)
                else:
                    axs[col].plot(theta, response)
                    axs[col].text(np.deg2rad(225),response.max()*1.5,'{}'.format(roi_index),fontsize=8)
            if n_rows>1:
                axs[row][col].tick_params(labelbottom=False)
                axs[row][col].tick_params(labelleft=False)
                axs[row][col].set_theta_direction(-1)
            else:
                axs[col].tick_params(labelbottom=False)
                axs[col].tick_params(labelleft=False)
                axs[col].set_theta_direction(-1)
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
            
def pre_process(h5_path, baseline=None, pre_onset=0):
    h5_data = h5py_read(h5_path)
    data = h5_data['rois']['rois_delta_fluor']
    trigger_idx = h5_data['vs']['trigger_index']
    vs_seq = h5_data['vs']['sequence']
    n_rep = h5_data['vs']['n_rep']
    # date_new_seq = datetime.datetime.strptime('20230726', '%Y%m%d')
    # # date_experiment = datetime.datetime.strptime(exp_date, '%Y%m%d')
    # if date_experiment>date_new_seq:
    #     vs_seq_idx = (vs_seq-1).astype(int)
    # else:
    #     vs_seq_idx = np.tile((vs_seq-1).astype(int),n_rep)   
    if vs_seq.min() == 1:
        vs_seq_idx = (vs_seq-1).astype(int)
    else:
        vs_seq_idx = vs_seq.astype(int)
    # print(vs_seq_idx)
    results = h5_data['vs']
    data_sorted = sorting(data, trigger_idx, vs_seq_idx, pre_onset=pre_onset)
    n_stim = data_sorted.shape[1]
    results['rois_area'] = h5_data['rois']['rois_area']
    results['rois_channel'] = h5_data['rois']['rois_channel']
    # results['rois_corr'] = h5_data['rois']['rois_corr']
    results['rois_pos'] = h5_data['rois']['rois_pos']
    n_rois = results['rois_area'].size
    results['n_rois'] = n_rois
    snr = np.zeros((n_stim,n_rois))
    for i_roi in range(n_rois):
        for i_stim in range(n_stim):
            snr[i_stim,i_roi] = cal_snr(data_sorted[:,i_stim,:-1,i_roi])
    results['snr'] = snr
    
    # this is added on 20230925
    idx_fin = int(h5_data['vs']['stim_duration']/h5_data['imaging']['frame_period'])
    peak_amp = find_peak(data_sorted[:idx_fin,:,:,:])
    peak_amp_rebound = find_peak(data_sorted[idx_fin:,:,:,:])
    results['peak_amp'] = peak_amp
    results['peak_amp_rebound'] = peak_amp_rebound
    # results['file_path'] = h5_path
    
    # change to average response 20231020
    # data_sorted: n_sample_per_stim, n_stim, n_rep+1, n_roi+1
    if baseline is None:
        # baseline = np.mean(data_sorted[-5:,:,:,:n_rois],axis=0) # the last 5 samples as baseline
        baseline_sample = np.mean(data_sorted[:5,:,:,:n_rois],axis=0) # the first 5 samples as baseline
        baseline_min = np.median(baseline_sample, axis=0)
        baseline = np.zeros_like(baseline_sample)
        for i in range(n_stim):
            baseline[i,:,:] = baseline_min

    data_sorted_new = data_sorted.copy()
    for i in range(data_sorted.shape[0]):
        # the behavior data not subtract baseline
        data_sorted_new[i,:,:,:n_rois] = (data_sorted[i,:,:,:n_rois]-baseline)/(baseline+1)
    results['data_sorted'] = data_sorted_new
    mean_amp = mean_resp(data_sorted_new[:idx_fin,:,:,:])
    mean_amp_rebound = mean_resp(data_sorted_new[idx_fin:,:,:,:n_rois]) # the behavior data not included
    results['mean_amp'] = mean_amp
    results['mean_amp_rebound'] = mean_amp_rebound
    results['sampling_rate'] = 1 / h5_data['imaging']['frame_period']
    
    return results

def idx_to_visual(rf_idx,para):
    x=rf_idx[0]
    y=rf_idx[1]
    visual_origin = np.array([para['stim_center'][0]-para['n_cols']/2*para['stim_size'],para['stim_center'][1]+para['n_rows']/2*para['stim_size']])
    rf_pos_deg = np.zeros(2)
    rf_pos_deg[0] = visual_origin[0]+(y+0.5)*para['stim_size']
    rf_pos_deg[1] = visual_origin[1]-(x+0.5)*para['stim_size']
    # x+=1
    # y+=1
    # rf_pos_deg = [(y-(para['n_cols']+1)/2)*para['size']+para['visual_center'][0],
    #                         -(x-(para['n_rows']+1)/2)*para['size']+para['visual_center'][1]
    return rf_pos_deg
    
def visual_to_idx(rf_deg,para):
    x_deg = rf_deg[0]
    y_deg = rf_deg[1]
    visual_origin = np.array([para['stim_center'][0]-para['n_cols']/2*para['stim_size'],para['stim_center'][1]+para['n_rows']/2*para['stim_size']])
    # print(visual_origin)
    rf_pos_idx = np.zeros(2)
    rf_pos_idx[0] = (visual_origin[1]-y_deg)/para['stim_size']-0.5
    rf_pos_idx[1] = (-visual_origin[0]+x_deg)/para['stim_size']-0.5 #[col,row] change to [row,col]
    return rf_pos_idx
    
def analysis_rf(results_rf,plot_results=False,plot_fitting=False):   
    # separate on and off subfields
    rf_row = results_rf['n_rows']
    rf_col = results_rf['n_cols']
    n_rois = results_rf['n_rois']
    stim_center = results_rf['stim_center']
    rf_on = np.zeros((rf_row, rf_col, n_rois))
    rf_off = np.zeros((rf_row, rf_col, n_rois))
    n_rf_stim = rf_row*rf_col  
    # changed from peak_amp to mean_amp on 20231023
    # peak_amp = results_rf['peak_amp']
    # for roi in range(n_rois):
    #     rf_on[:, :, roi] = peak_amp[0:n_rf_stim,-1, roi].reshape((rf_col, rf_row)).T # for on stimulus
    #     rf_off[:, :, roi] = peak_amp[n_rf_stim:n_rf_stim*2,-1,roi].reshape((rf_col, rf_row)).T # for off stimulus   
    mean_amp = results_rf['mean_amp']
    for roi in range(n_rois):
        rf_on[:, :, roi] = mean_amp[0:n_rf_stim,-1, roi].reshape((rf_col, rf_row)).T # for on stimulus
        rf_off[:, :, roi] = mean_amp[n_rf_stim:n_rf_stim*2,-1,roi].reshape((rf_col, rf_row)).T # for off stimulus   
    results_rf['rf_on'] = rf_on
    results_rf['rf_off'] = rf_off
    rf_on_max = np.max(rf_on,axis=(0,1))
    rf_off_max = np.max(rf_off,axis=(0,1))
    rf_csi = (rf_on_max - rf_off_max)/(rf_on_max + rf_off_max)
    results_rf['rf_csi'] = rf_csi
    interp_num = 5
    rf_fit_list = []
    if 'rf_area_fit' not in results_rf:
        rf_pos_deg = np.zeros((2,n_rois))
        rf_area = np.zeros(n_rois)
        for i in tqdm(range(n_rois)):
            if rf_csi[i]>0:
                rf_fit_para, rf_2d_fit = rf_process(rf_on[:,:,i],visual_center=stim_center,interp_num=interp_num,plot=plot_fitting)
            else:
                rf_fit_para, rf_2d_fit = rf_process(rf_off[:,:,i],visual_center=stim_center,interp_num=interp_num,plot=plot_fitting)
            rf_fit_list.append(rf_2d_fit)
            rf_pos_deg[0,i] = rf_fit_para['x0']
            rf_pos_deg[1,i] = rf_fit_para['y0']
            rf_area[i] = rf_fit_para['area']
        results_rf['rf_pos_deg_fit'] = rf_pos_deg
        results_rf['rf_area_fit'] = rf_area
        results_rf['rf_2d_fit'] = np.asarray(rf_fit_list)
            
    if 'rf_pos_deg_cal' not in results_rf:
        rf_pos_deg_cal = np.zeros((2,n_rois))
        for i in range(n_rois):
            if rf_csi[i]>0:
                rf_2d = np.copy(rf_on[:,:,i])
            else:
                rf_2d = np.copy(rf_off[:,:,i])
            n_row,n_col = rf_2d.shape
            rf_2d = rf_2d - np.mean(rf_2d)
            rf_2d = np.where(rf_2d < (np.max(rf_2d)/3), 0, rf_2d)
            x,y=scipy.ndimage.center_of_mass(rf_2d)
            rf_pos_deg_cal[:,i] = idx_to_visual([x,y],results_rf)
            
        results_rf['rf_pos_deg_cal'] = rf_pos_deg_cal
    
    if plot_results:
        folder_path = os.path.dirname(results_rf['file_path'])
        plot_rfs(results_rf['rf_on'],n_cols=10)
        save_fig(folder_path+'/rf_on')
        plot_rfs(results_rf['rf_off'],n_cols=10)
        save_fig(folder_path+'/rf_off')       
    return results_rf

def analysis_rf_identical(results_rf,plot_results=False,plot_fitting=False):   
    # this do not separate on and off subfields
    rf_row = results_rf['n_rows']
    rf_col = results_rf['n_cols']
    n_rois = results_rf['n_rois']
    stim_center = results_rf['monitor_center']
    rf_on = np.zeros((rf_row, rf_col, n_rois))
    n_rf_stim = rf_row*rf_col  

    mean_amp = results_rf['mean_amp']
    for roi in range(n_rois):
        rf_on[:, :, roi] = mean_amp[:, -1, roi].reshape((rf_col, rf_row)).T # for on stimulus
 
    results_rf['rf_on'] = rf_on

    interp_num = 5
    rf_fit_list = []
    if 'rf_area_fit' not in results_rf:
        rf_pos_deg = np.zeros((2,n_rois))
        rf_area = np.zeros(n_rois)
        for i in tqdm(range(n_rois)):
            rf_fit_para, rf_2d_fit = rf_process(rf_on[:,:,i],visual_center=stim_center,interp_num=interp_num,plot=plot_fitting)

            rf_fit_list.append(rf_2d_fit)
            rf_pos_deg[0,i] = rf_fit_para['x0']
            rf_pos_deg[1,i] = rf_fit_para['y0']
            rf_area[i] = rf_fit_para['area']
        results_rf['rf_pos_deg_fit'] = rf_pos_deg
        results_rf['rf_area_fit'] = rf_area
        results_rf['rf_2d_fit'] = np.asarray(rf_fit_list)
            
    if 'rf_pos_deg_cal' not in results_rf:
        rf_pos_deg_cal = np.zeros((2,n_rois))
        for i in range(n_rois):
            rf_2d = np.copy(rf_on[:,:,i])
            n_row,n_col = rf_2d.shape
            rf_2d = rf_2d - np.mean(rf_2d)
            rf_2d = np.where (rf_2d < (np.max(rf_2d)/3), 0, rf_2d)
            x,y=scipy.ndimage.center_of_mass(rf_2d)
            rf_pos_deg_cal[:,i] = idx_to_visual([x,y],results_rf)
            
        results_rf['rf_pos_deg_cal'] = rf_pos_deg_cal
        
    return results_rf

def analysis_mb(results_mb, snr_thr=0.5, plot_results=False):
    n_rois = results_mb['n_rois']
    peak_amp = results_mb['peak_amp']
    mean_amp = results_mb['mean_amp'] 
    rois_pos = results_mb['rois_pos']
    pref_ori = np.zeros(n_rois)
    osi = np.zeros(n_rois)
    pref_dir = np.zeros(n_rois)
    dsi = np.zeros(n_rois)
    for i in range(n_rois):
        pref_ori[i], osi[i], pref_dir[i], dsi[i] = ori_dir(mean_amp[:, -1, i]) # changed to mean_amp on 20231023
    results_mb['pref_ori'] = pref_ori    
    results_mb['pref_dir'] = pref_dir
    results_mb['osi'] = osi 
    results_mb['dsi'] = dsi

    results_mb['speed_mm'] = 2 * results_mb['distance'] * math.tan(math.radians(results_mb['bar_speed'] / 2 ))
    rf_mb = np.zeros((2,n_rois))
    for i in range(n_rois):
        rf_mb[:,i] = cal_rf_mb(results_mb['data_sorted'][:,:,:,i],snr_th=snr_thr,para=results_mb)
    results_mb['rf_mb'] = rf_mb
    
    if plot_results:
        folder_path = os.path.dirname(results_mb['file_path'])
        arrow_plot(pref_dir,rois_pos)
        save_fig(folder_path+'/arrows')
        plot_polar_graphs(mean_amp[:,-1,:])
        save_fig(folder_path+'/polar_graph')
    return results_mb

def analysis_md(results_md,snr_thr=0.5,plot_results=False):
    n_rois = results_md['n_rois']
    peak_amp = results_md['peak_amp']
    mean_amp = results_md['mean_amp']
    rois_pos = results_md['rois_pos']
    pref_dir = np.zeros(n_rois)
    dsi = np.zeros(n_rois)
    for i in range(n_rois):
        _, _, pref_dir[i], dsi[i] = ori_dir(mean_amp[:, -1, i]) # changed to mean_amp on 20231023
    results_md['pref_dir'] = pref_dir 
    results_md['dsi'] = dsi
    
    if plot_results:
        folder_path = os.path.dirname(results_md['file_path'])
        arrow_plot(pref_dir,rois_pos)
        save_fig(folder_path+'/arrows')
        plot_polar_graphs(mean_amp[:,-1,:])
        save_fig(folder_path+'/polar_graph')
    return results_md

def analysis_sg0(results, para,snr_thr=0.5, plot_results=False):
    n_rois = results['n_rois']
    mean_amp = results['mean_amp'] 
    pref_ori = np.zeros(n_rois)
    osi = np.zeros(n_rois)
    for i in range(n_rois):
        pref_ori[i], osi[i] = cal_ori(mean_amp[:, -1, i]) # changed to mean_amp on 20231023
    results['pref_ori'] = pref_ori    
    results['osi'] = osi 
    
    return results

def analysis_sg1(results,plot_results=False):   
    n_rows = results['n_rows']
    n_cols = results['n_cols']
    n_rois = results['n_rois']
    peak_amp = results['peak_amp']
    mean_amp = results['mean_amp']
    sg1_0 = np.zeros((n_rows, n_cols, n_rois))
    sg1_90 = np.zeros((n_rows, n_cols, n_rois))
    n_stim = n_rows*n_cols
    for roi in range(n_rois):
        sg1_0[:, :, roi] = mean_amp[0:n_stim,-1, roi].reshape((n_cols, n_rows)).T # for on stimulus # changed to mean_amp on 20231023
        sg1_90[:, :, roi] = mean_amp[n_stim:n_stim*2,-1,roi].reshape((n_cols, n_rows)).T # for off stimulus   # changed to mean_amp on 20231023
    results['sg1_0'] = sg1_0
    results['sg1_90'] = sg1_90
    if plot_results:
        folder_path = os.path.dirname(results['file_path'])
        plot_rfs(results['sg1_0'],n_cols=10)
        save_fig(folder_path+'/sg1_0')
        plot_rfs(results['sg1_90'],n_cols=10)
        save_fig(folder_path+'/sg1_90')
    return results

def analysis_sg2(results,plot_results=False):   
    n_rois = results['n_rois']
    n_stim = results['n_stim']
    n_rows = 2
    n_cols = n_stim//n_rows
    # peak_amp = results['peak_amp']
    mean_amp = results['mean_amp']
    sg2_0_90 = np.zeros((n_rows, n_cols, n_rois))
    for roi in range(n_rois):
        sg2_0_90[:, :, roi] = mean_amp[:,-1,roi].reshape((n_rows,n_cols))# changed to mean_amp on 20231023
    results['sg2_0_90'] = sg2_0_90
    if plot_results:
        folder_path = os.path.dirname(results['file_path'])
        plot_rfs(results['sg2_0_90'],n_cols=10)
        save_fig(folder_path+'/sg2_0_90')
    return results

def analysis_sg3(results,plot_results=False):   
    n_rois = results['n_rois']
    n_stim = results['n_stim']
    n_rows = 2
    n_cols = n_stim//n_rows
    peak_amp = results['peak_amp']
    mean_amp = results['mean_amp']
    sg3_0_90 = np.zeros((n_rows, n_cols, n_rois))
    for roi in range(n_rois):
        sg3_0_90[:, :, roi] = mean_amp[:,-1,roi].reshape((n_rows,n_cols))# changed to mean_amp on 20231023
    results['sg3_0_90'] = sg3_0_90
    if plot_results:
        folder_path = os.path.dirname(results['file_path'])
        plot_rfs(results['sg3_0_90'],n_cols=10)
        save_fig(folder_path+'/sg3_0_90')
    return results

def analysis_smd1(results,plot_results=False):
    n_rows = results['n_rows']
    n_cols = results['n_cols']
    n_rois = results['n_rois']
    peak_amp = results['peak_amp']
    mean_amp = results['mean_amp']
    smd1_0 = np.zeros((n_rows, n_cols, n_rois))
    smd1_90 = np.zeros((n_rows, n_cols, n_rois))
    smd1_180 = np.zeros((n_rows, n_cols, n_rois))
    smd1_270 = np.zeros((n_rows, n_cols, n_rois))
    n_stim = n_rows*n_cols
    for roi in range(n_rois):
        smd1_0[:, :, roi] = mean_amp[0:n_stim,-1, roi].reshape((n_cols, n_rows)).T  # changed to mean_amp on 20231023
        smd1_90[:, :, roi] = mean_amp[n_stim:n_stim*2,-1,roi].reshape((n_cols, n_rows)).T  # changed to mean_amp on 20231023
        smd1_180[:, :, roi] = mean_amp[n_stim*2:n_stim*3,-1, roi].reshape((n_cols, n_rows)).T # changed to mean_amp on 20231023
        smd1_270[:, :, roi] = mean_amp[n_stim*3:,-1,roi].reshape((n_cols, n_rows)).T  # changed to mean_amp on 20231023
    results['smd1_0'] = smd1_0
    results['smd1_90'] = smd1_90
    results['smd1_180'] = smd1_180
    results['smd1_270'] = smd1_270
    if plot_results:
        folder_path = os.path.dirname(results['file_path'])
        plot_rfs(results['smd1_0'],n_cols=10)
        save_fig(folder_path+'/smd1_0')
        plot_rfs(results['smd1_90'],n_cols=10)
        save_fig(folder_path+'/smd1_90')
        folder_path = os.path.dirname(results['file_path'])
        plot_rfs(results['smd1_180'],n_cols=10)
        save_fig(folder_path+'/smd1_180')
        plot_rfs(results['smd1_270'],n_cols=10)
        save_fig(folder_path+'/smd1_270')
        
    return results

def analysis_smd2(results,plot_results=False):   
    n_rois = results['n_rois']
    n_stim = results['n_stim']
    n_rows = 4
    n_cols = n_stim//n_rows
    peak_amp = results['peak_amp']
    mean_amp = results['mean_amp']
    smd2 = np.zeros((n_rows, n_cols, n_rois))
    for roi in range(n_rois):
        smd2[:, :, roi] = mean_amp[:,-1,roi].reshape((n_rows,n_cols))# changed to mean_amp on 20231023
    results['smd2'] = smd2
    if plot_results:
        folder_path = os.path.dirname(results['file_path'])
        plot_rfs(results['smd2'],n_cols=10)
        save_fig(folder_path+'/smd2')
    return results

def analysis_smd3(results,plot_results=False):   
    n_rois = results['n_rois']
    n_stim = results['n_stim']
    n_rows = 4
    n_cols = n_stim//n_rows
    peak_amp = results['peak_amp']
    mean_amp = results['mean_amp']
    smd3 = np.zeros((n_rows, n_cols, n_rois))
    for roi in range(n_rois):
        smd3[:, :, roi] = mean_amp[:,-1,roi].reshape((n_rows,n_cols))# changed to mean_amp on 20231023
    results['smd3'] = smd3
    if plot_results:
        folder_path = os.path.dirname(results['file_path'])
        plot_rfs(results['smd3'],n_cols=10)
        save_fig(folder_path+'/smd3')
    return results

def analysis_smg0(results,snr_thr=0.5,plot_results=False):
    n_rois = results['n_rois']
    # peak_amp = results_smg0['peak_amp']
    mean_amp = results['mean_amp']
    # rois_pos = results_smg0['rois_pos']
    pref_oir = np.zeros(n_rois)
    osi = np.zeros(n_rois)
    pref_dir = np.zeros(n_rois)
    dsi = np.zeros(n_rois)
    for i in range(n_rois):
        pref_oir[i], osi[i], pref_dir[i], dsi[i] = ori_dir(mean_amp[:, -1, i]) # changed to mean_amp on 20231023
    results['pref_oir'] = pref_oir
    results['osi'] = osi
    results['pref_dir'] = pref_dir
    results['dsi'] = dsi

    return results

def analysis_smg1(results, plot_results=False):
    n_rows = results['n_rows']
    n_cols = results['n_cols']
    n_rois = results['n_rois']
    peak_amp = results['peak_amp']
    mean_amp = results['mean_amp']
    smg1_0 = np.zeros((n_rows, n_cols, n_rois))
    smg1_90 = np.zeros((n_rows, n_cols, n_rois))
    smg1_180 = np.zeros((n_rows, n_cols, n_rois))
    smg1_270 = np.zeros((n_rows, n_cols, n_rois))
    n_stim = n_rows*n_cols
    for roi in range(n_rois):
        smg1_0[:, :, roi] = mean_amp[0:n_stim,-1, roi].reshape((n_cols, n_rows)).T  # changed to mean_amp on 20231023
        smg1_90[:, :, roi] = mean_amp[n_stim:n_stim*2,-1,roi].reshape((n_cols, n_rows)).T  # changed to mean_amp on 20231023
        smg1_180[:, :, roi] = mean_amp[n_stim*2:n_stim*3,-1, roi].reshape((n_cols, n_rows)).T # changed to mean_amp on 20231023
        smg1_270[:, :, roi] = mean_amp[n_stim*3:,-1,roi].reshape((n_cols, n_rows)).T  # changed to mean_amp on 20231023
    results['smg1_0'] = smg1_0
    results['smg1_90'] = smg1_90
    results['smg1_180'] = smg1_180
    results['smg1_270'] = smg1_270

    return results

def analysis_smg2(results, plot_results=False):
    n_rows = results['n_rows']
    n_stim = results['n_stim']
    n_cols = n_stim//n_rows
    n_rois = results['n_rois']
    peak_amp = results['peak_amp']
    mean_amp = results['mean_amp']
    smg2 = np.zeros((n_rows, n_cols, n_rois))
    for roi in range(n_rois):
        smg2[:, :, roi] = mean_amp[:, -1, roi].reshape((n_cols, n_rows)).T  # changed to mean_amp on 20231023
    results['smg2'] = smg2

    return results

def analysis_smg3(results, plot_results=False):
    n_rows = results['n_rows']
    n_stim = results['n_stim']
    n_cols = n_stim//n_rows
    n_rois = results['n_rois']
    peak_amp = results['peak_amp']
    mean_amp = results['mean_amp']
    smg3 = np.zeros((n_rows, n_cols, n_rois))
    for roi in range(n_rois):
        smg3[:, :, roi] = mean_amp[:, -1, roi].reshape((n_cols, n_rows)).T  # changed to mean_amp on 20231023
    results['smg3'] = smg3

    return results

def plot_selected_rois(roi_pos, selected_index, label=[], title=None):
    lim_max = 512
    dots_size = 1
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')
    ax.set_xlim(0,lim_max)
    ax.set_ylim(0,lim_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    x = roi_pos[:, 0]
    y = roi_pos[:, 1]
    ax.scatter(x, y, s=dots_size)
    ax.invert_yaxis()
    if len(label) == 0:
        for r in selected_index:
            xy = (x[r], y[r])
            xytext =  (x[r]-7, y[r]-2.5)
            text = ax.annotate('{}'.format(r), xy=xy, xytext=xytext, fontsize=8)
            text.set_alpha(0.7)
    else:
        for i,r in enumerate(selected_index):
            xy = (x[r], y[r])
            xytext =  (x[r]-7, y[r]-2.5)
            text = ax.annotate('{}'.format(label[i]), xy=xy, xytext=xytext, fontsize=8)
            text.set_alpha(0.7)
    plt.suptitle(title)
    
def plot_example_rois(roi_pos, selected_index,figsize=(10,10),title=None):
    lim_max = 512
    dots_size = 1
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    ax.set_xlim(0,lim_max)
    ax.set_ylim(0,lim_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    x = roi_pos[:, 0]
    y = roi_pos[:, 1]
    ax.scatter(x, y, s=dots_size)
    ax.invert_yaxis()
    r = selected_index
    xy = (x[r], y[r])
    ax.scatter(x[r], y[r], s=dots_size*50, marker='.', c='red')
    xytext =  (x[r]-7, y[r]-2.5)
    text = ax.annotate('{}'.format(r), xy=xy, xytext=xytext, fontsize=8)
    text.set_alpha(0.7)
    plt.suptitle(title)

def plot_response(data, n_rows, window=[0, 10], scale_bar=True, title=None):
    '''
    data: 3D array, (n_frames, n_stim, n_repeats)
    '''
    xlim_min = 0
    xlim_max = data.shape[0]
    # n_stim = data.shape[1]
    n_cols = int(data.shape[1] / n_rows)
    ylim_min = np.min(data)
    ylim_max = np.max(data)
    print('ylim_max - ylim_min = {}'.format(ylim_max-ylim_min))

    fig = plt.figure(layout='constrained', figsize=(n_cols, n_rows))
    subfigs = fig.subfigures(1, n_cols, wspace=0.07)
    for c in range(n_cols):
        axis = subfigs[c].subplots(n_rows, 1)
        for i in range(n_rows):
            mean = np.mean(data[:,i+c*n_rows,:], axis=1)
            y1 = mean + np.std(data[:,i+c*n_rows,:], axis=1)
            y2 = mean - np.std(data[:,i+c*n_rows,:], axis=1)
            axis[i].fill_between(np.arange(xlim_max), y1, y2, color='black', alpha=0.3)
            axis[i].plot(mean, color='black')
            axis[i].set_xlim(xlim_min, xlim_max)
            axis[i].set_ylim(0.8*ylim_min, ylim_max)
            axis[i].set_xticks([])
            axis[i].set_yticks([])
            axis[i].spines['top'].set_visible(False)
            axis[i].spines['right'].set_visible(False)
            axis[i].spines['left'].set_visible(False)
            axis[i].spines['bottom'].set_visible(False)
            axis[i].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')
            # axis[i].set_ylabel('{}'.format(i+1))

            # Add horizontal and vertical scale bars with labels
            if (i == n_rows-1) and (c == n_cols-1) and scale_bar:
                top = 0.99
                right = 0.99
                hline_length = 0.33
                vline_length = 0.3 / (ylim_max - ylim_min)
                axis[i].axhline(y = ylim_min + (ylim_max-ylim_min)*top, xmin=right-hline_length, xmax=right, color='black', linewidth=1)
                # axis[i].text(0.15, ylim_min - 0.1, '10 units', fontsize=12)
                axis[i].axvline(x = right*xlim_max, ymin=top-vline_length, ymax=top, color='black', linewidth=1)
                # axis[i].text(xlim_min - 0.5, 0.15, '0 ms', fontsize=12)

    if title is not None:
        plt.suptitle(title)

def plot_response_polar(data, window=[0, 50], title=None):

    xlim_min = 0
    xlim_max = data.shape[0]
    ylim_min = np.min(data)
    ylim_max = np.max(data)

    n_stim = data.shape[1]

    # for plot the directioin preference
    angles = np.arange(0, 361, 30)
    theta = np.deg2rad(angles)

    fig = plt.figure(layout='constrained', figsize=(8, 6))
    subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[1, 2])

    axis = subfigs[0].subplots(n_stim, 1)
    for i in range(n_stim):

        mean = np.mean(data[:,i,:], axis=1)
        y1 = mean + np.std(data[:,i,:], axis=1)
        y2 = mean - np.std(data[:,i,:], axis=1)
        axis[i].fill_between(np.arange(xlim_max), y1, y2, alpha=0.3)
        axis[i].plot(mean)
        axis[i].set_xlim(xlim_min, xlim_max)
        axis[i].set_ylim(ylim_min, ylim_max)
        axis[i].set_xticks([])
        axis[i].set_yticks([])
        axis[i].spines['top'].set_visible(False)
        axis[i].spines['right'].set_visible(False)
        axis[i].spines['left'].set_visible(False)
        axis[i].spines['bottom'].set_visible(False)
        axis[i].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')
        axis[i].set_ylabel('{}'.format(angles[i]))

    ax_polar = subfigs[1].add_subplot(projection='polar')
    
    response = find_peak(data=np.mean(data, axis=2))
    response = np.append(response, response[0]) # append a element for plot a closed response
    ax_polar.plot(theta, response)
    ax_polar.set_theta_direction(-1)
    # ax_polar.tick_params(labelbottom=False)
    # ax_polar.tick_params(labelleft=False)
    plt.suptitle(title)
    
def plot_mb_traces_example(data, figsize=(3,6),title=None):
    xlim_min = 0
    xlim_max = data.shape[0]
    ylim_min = np.min(data)
    ylim_max = np.max(data)

    n_stim = data.shape[1]

    fig = plt.figure(layout='constrained', figsize=figsize)

    axis = fig.subplots(n_stim, 1)
    for i in range(n_stim):

        mean = np.mean(data[:,i,:], axis=1)
        y1 = mean + np.std(data[:,i,:], axis=1)
        y2 = mean - np.std(data[:,i,:], axis=1)
        axis[i].fill_between(np.arange(xlim_max), y1, y2, alpha=0.3,color='gray')
        axis[i].plot(mean,'black')
        axis[i].set_xlim(xlim_min, xlim_max)
        axis[i].set_ylim(ylim_min, ylim_max)
        axis[i].set_xticks([])
        axis[i].set_yticks([])
        axis[i].spines['top'].set_visible(False)
        axis[i].spines['right'].set_visible(False)
        axis[i].spines['left'].set_visible(False)
        # axis[i].hlines(y=baseline_stim2[i, rep, roi_index], xmin=0, xmax=5, color='r')
        # axis[i].set_ylabel('{}'.format(angles[i]))

    plt.suptitle(title)

def plot_response_RF(data, window=[0, 10], title=None):
    '''
    plot on and off response of RF
    '''
    nrows = 4
    ncols = 6

    ylim_max = data.max()
    ylim_min = data.min()

    data_on = data[:, 0:24, :]
    data_off = data[:, 24:48, :]

    temp = find_peak(data=np.mean(data, axis=2))
    vmin = np.min(temp)
    vmax = np.max(temp)

    fig = plt.figure(layout='constrained', figsize=(10, 10))
    subfigs = fig.subfigures(2, 2, wspace=0.07)
    axis_trace = subfigs[0, 0].subplots(nrows, ncols)
    for col in range(ncols):
        for row in range(nrows):
            location_index = row + col*nrows
            mean = np.mean(data_on[:,location_index,:], axis=1)
            y1 = mean + np.std(data_on[:,location_index,:], axis=1)
            y2 = mean - np.std(data_on[:,location_index,:], axis=1)
            axis_trace[row][col].fill_between(np.arange(data.shape[0]), y1, y2, color='black', alpha=0.3)
            axis_trace[row][col].plot(mean, color='black')
            axis_trace[row][col].spines['top'].set_visible(False)
            axis_trace[row][col].spines['right'].set_visible(False)
            axis_trace[row][col].spines['left'].set_visible(False)
            axis_trace[row][col].spines['bottom'].set_visible(False)
            axis_trace[row][col].set_ylim(ylim_min, ylim_max)
            axis_trace[row][col].set_xticks([])
            axis_trace[row][col].set_yticks([])
            # axis_trace[row][col].set_xlabel('{}'.format(location_index))
            axis_trace[row][col].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')

    axis_right = subfigs[0, 1].subplots()
    # for plot the RF
    RF_on = np.zeros((4, 6))
    temp = find_peak(data=np.mean(data_on, axis=2))
    RF_on[:, :] = temp.reshape((6, 4)).T # for on stimulus
    axis_right.imshow(RF_on, cmap='gray', vmin=vmin, vmax=vmax)
    # axis_right.set_title('RF_on')
    axis_right.set_xticks([])
    axis_right.set_yticks([])
    
    axis_trace = subfigs[1, 0].subplots(nrows, ncols)
    for col in range(ncols):
        for row in range(nrows):
            location_index = row + col*nrows
            mean = np.mean(data_off[:,location_index,:], axis=1)
            y1 = mean + np.std(data_off[:,location_index,:], axis=1)
            y2 = mean - np.std(data_off[:,location_index,:], axis=1)
            axis_trace[row][col].fill_between(np.arange(data.shape[0]), y1, y2, color='black', alpha=0.3)
            axis_trace[row][col].plot(mean, color='black')
            axis_trace[row][col].spines['top'].set_visible(False)
            axis_trace[row][col].spines['right'].set_visible(False)
            axis_trace[row][col].spines['left'].set_visible(False)
            axis_trace[row][col].spines['bottom'].set_visible(False)
            axis_trace[row][col].set_ylim(ylim_min, ylim_max)
            axis_trace[row][col].set_xticks([])
            axis_trace[row][col].set_yticks([])
            # axis_trace[row][col].set_xlabel('{}'.format(location_index))
            axis_trace[row][col].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')

    axis_right = subfigs[1, 1].subplots()
    # for plot the RF
    RF_off = np.zeros((4, 6))
    temp = find_peak(data=np.mean(data_off, axis=2))
    RF_off[:, :] = temp.reshape((6, 4)).T # for on stimulus
    axis_right.imshow(RF_off, cmap='gray', vmin=vmin, vmax=vmax)
    # axis_right.set_title('RF_off')
    axis_right.set_xticks([])
    axis_right.set_yticks([])
    plt.suptitle(title, fontsize=20)

def plot_response_RF_single(data, window=[0, 10], scale_bar=True, title=None):
    '''
    plot on or off response of RF
    '''
    nrows = 4
    ncols = 6
    xlim_min = 0
    xlim_max = data.shape[0]
    ylim_min = data.min()
    ylim_max = data.max()
    print('ylim_max - ylim_min = {}'.format(ylim_max-ylim_min))

    temp = find_peak(data=np.mean(data, axis=2))
    vmin = np.min(temp)
    vmax = np.max(temp)
    width = 6
    hight = 10

    fig = plt.figure(layout='constrained', figsize=(width, hight))
    subfigs = fig.subfigures(nrows=2, ncols=1, wspace=0.07)
    axis_trace = subfigs[0].subplots(nrows, ncols)
    for col in range(ncols):
        for row in range(nrows):
            location_index = row + col*nrows
            mean = np.mean(data[:,location_index,:], axis=1)
            y1 = mean + np.std(data[:,location_index,:], axis=1)
            y2 = mean - np.std(data[:,location_index,:], axis=1)
            axis_trace[row][col].fill_between(np.arange(xlim_max), y1, y2, color='black', alpha=0.3)
            axis_trace[row][col].plot(mean, color='black')
            axis_trace[row][col].spines['top'].set_visible(False)
            axis_trace[row][col].spines['right'].set_visible(False)
            axis_trace[row][col].spines['left'].set_visible(False)
            axis_trace[row][col].spines['bottom'].set_visible(False)
            axis_trace[row][col].set_xlim(xlim_min, xlim_max)
            axis_trace[row][col].set_ylim(ylim_min, ylim_max)
            axis_trace[row][col].set_xticks([])
            axis_trace[row][col].set_yticks([])
            # axis_trace[row][col].set_xlabel('{}'.format(location_index))
            axis_trace[row][col].hlines(y=0, xmin=window[0], xmax=window[1], color='r', linewidth=1, linestyle='--')

    # Add horizontal and vertical scale bars
    if scale_bar:
        top = 0.99
        right = 0.99
        hline_length = 0.2
        vline_length = 0.3 / (ylim_max - ylim_min)
        axis_trace[row][col].axhline(y = ylim_min + (ylim_max-ylim_min)*top, xmin=right-hline_length, xmax=right, color='black', linewidth=1)

        axis_trace[row][col].axvline(x = right*xlim_max, ymin=top-vline_length, ymax=top, color='black', linewidth=1)

    axis_bottom = subfigs[1].subplots()
    # for plot the RF
    RF_on = np.zeros((4, 6))
    temp = find_peak(data=np.mean(data, axis=2))
    RF_on[:, :] = temp.reshape((6, 4)).T # for on stimulus
    axis_bottom.imshow(RF_on, cmap='gray', vmin=vmin, vmax=vmax)
    # axis_bottom.set_title('RF_on')
    axis_bottom.set_xticks([])
    axis_bottom.set_yticks([])

    plt.suptitle(title)

def plot_RF_traces_example(data, figsize=(6,3),title=None):

    nrows = 4
    ncols = 6

    ylim_max = data.max()
    ylim_min = data.min()

    data_on = data[:, 0:24, :]
    data_off = data[:, 24:48, :]

    temp = find_peak(data=np.mean(data, axis=2))
    vmin = np.min(temp)
    vmax = np.max(temp)

    fig = plt.figure(layout='constrained', figsize=figsize)
    subfigs = fig.subfigures(1, 2, wspace=0.07)
    axis_trace = subfigs[0].subplots(nrows, ncols)
    for col in range(ncols):
        for row in range(nrows):
            location_index = row + col*nrows
            mean = np.mean(data_on[:,location_index,:], axis=1)
            y1 = mean + np.std(data_on[:,location_index,:], axis=1)
            y2 = mean - np.std(data_on[:,location_index,:], axis=1)
            axis_trace[row][col].fill_between(np.arange(data.shape[0]), y1, y2, alpha=0.3,color='gray')
            axis_trace[row][col].plot(mean,color='black')
            axis_trace[row][col].spines['top'].set_visible(False)
            axis_trace[row][col].spines['right'].set_visible(False)
            axis_trace[row][col].spines['left'].set_visible(False)
            axis_trace[row][col].spines['bottom'].set_visible(False)
            axis_trace[row][col].set_ylim(ylim_min, ylim_max)
            axis_trace[row][col].set_xticks([])
            axis_trace[row][col].set_yticks([])
            # axis_trace[row][col].set_xlabel('{}'.format(location_index))
            axis_trace[row][col].hlines(y=0, xmin=0, xmax=20, color='red', linewidth=0.5, linestyle='--')
    
    axis_trace = subfigs[1].subplots(nrows, ncols)
    for col in range(ncols):
        for row in range(nrows):
            location_index = row + col*nrows
            mean = np.mean(data_off[:,location_index,:], axis=1)
            y1 = mean + np.std(data_off[:,location_index,:], axis=1)
            y2 = mean - np.std(data_off[:,location_index,:], axis=1)
            axis_trace[row][col].fill_between(np.arange(data.shape[0]), y1, y2, alpha=0.3,color='gray')
            axis_trace[row][col].plot(mean,color='black')
            axis_trace[row][col].spines['top'].set_visible(False)
            axis_trace[row][col].spines['right'].set_visible(False)
            axis_trace[row][col].spines['left'].set_visible(False)
            axis_trace[row][col].spines['bottom'].set_visible(False)
            axis_trace[row][col].set_ylim(ylim_min, ylim_max)
            axis_trace[row][col].set_xticks([])
            axis_trace[row][col].set_yticks([])
            # axis_trace[row][col].set_xlabel('{}'.format(location_index))
            axis_trace[row][col].hlines(y=0, xmin=0, xmax=20, color='red', linewidth=0.5, linestyle='--')
    plt.suptitle(title)
    
    
def plot_single_trace(data,ylim,figsize=(4,3),title=None):


    ylim_min = ylim[0]
    ylim_max = ylim[1]

    fig,ax = plt.subplots(1,1,figsize=figsize)
    data_mean = np.mean(data, axis=1)
    y1 = data_mean + np.std(data, axis=1)
    y2 = data_mean - np.std(data, axis=1)
    ax.fill_between(np.arange(data.shape[0]), y1, y2, alpha=0.3,color='gray')
    ax.plot(data_mean,color='black')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylim(ylim_min, ylim_max)
    ax.set_xticks([])
    ax.set_yticks([])
    baseline = np.sort(data_mean)[:3].mean()
    print(np.max(data_mean)-baseline)
    ax.hlines(y=baseline, xmin=0, xmax=data.shape[0], color='red', linewidth=0.5, linestyle='--')

    plt.suptitle(title)

def cal_rf_mb(data, snr_th, para):
    '''
    Parameters
    ----------
    data : numpy array: n_sample,n_stim,n_rep
        
    snr_th : TYPE
        DESCRIPTION.
    para : TYPE
        DESCRIPTION.

    Returns
    -------
    rf_pos : TYPE
        rf position in degree.
    '''
    [n_sample,n_stim,n_rep] = np.shape(data)
    snr = np.zeros(n_stim)
    for i in range(n_stim):
        snr[i] = cal_snr(np.reshape(data[:,i,:],(-1,n_rep)))
    
    dist = para['distance']
    azi = para['monitor_center'][0]
    ele = para['monitor_center'][1]
    w = para['width']
    h = para['height']
    v = para['speed_mm'] #139.7 mm per sec
    sr = para['sampling_rate']
    angle = np.arange(12)*np.deg2rad(30) # 0
    travel_mm_0_90 = np.sqrt(w**2+h**2)*np.sin(angle[:4]+np.arctan(w/h))
    travel_mm = np.zeros(n_stim)
    travel_mm[:4] = travel_mm_0_90
    travel_mm[4:6] = travel_mm_0_90[1:3][::-1]
    travel_mm[6:] = travel_mm[:6]
    latency_lim = travel_mm/v   
    data_mean = np.mean(data,axis=2)
    win = np.round(sr*0.5) # 0.5 sec moving window
    latency_peak = np.argmax(data_mean,axis=0)/sr
    idx = np.where(snr>snr_th)[0]
    if idx.size <=4:
        rf_deg = np.asarray([np.nan,np.nan])
    else:
        f = np.concatenate((-np.ones(3),np.ones(6),-np.ones(3)),axis=None)
        g = np.concatenate((np.ones(6),-np.ones(6)),axis=None)
        d = travel_mm - latency_peak*v
        
        # A = np.array([-np.cos(angle(idx)), -np.sin(angle(idx))])
        # B = d[idx]
        # X = np.linalg.solve(A,B)
        
        ## calculate the coordinates from four corners at 12 directions
        x0 = f*w/2+latency_peak*v*np.cos(angle)
        y0 = g*h/2-latency_peak*v*np.sin(angle)

#       define the linear equation
        # (y-y0)/(x-x0)==np.tan(angle) 
        
        A = np.stack((np.cos(angle[idx]), -np.sin(angle[idx])),axis=1)
        B = x0[idx]*np.cos(angle[idx]) -y0[idx]*np.sin(angle[idx])
        # print(B.shape)
        rf_mm,_,_,_ = np.linalg.lstsq(A,B)
        if np.abs(rf_mm)[0] > w/2 or np.abs(rf_mm)[1] > h/2:
            rf_mm = np.nan
        rf_deg = np.rad2deg(np.arctan(rf_mm/dist))+np.array([azi,ele])
        
    return rf_deg

def norm_x(x,formula = 0):
    '''
    normalize each row

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    formula : TYPE, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    y : TYPE
        DESCRIPTION.

    '''
    y = np.copy(x)
    if x.ndim == 2:
        for i,row in enumerate(y):
            if formula == 0:
                temp = row - min(row)
                y[i] = temp/max(temp)
            else:
                y[i] = row/max([abs(max(row)),abs(min(row))])  
    elif x.ndim == 1:
        if formula == 0:
            temp = x-min(x)
            y = temp/max(temp)
    return y

def plot_exmaples(rf,snr_ranking_num,cmap,norm,plot_figure=True):
    snr_rf_max = np.max(rf['snr'],axis=0)
    snr_rf_max_idx = np.argmax(rf['snr'],axis=0)
    roi_rf_sorted = np.argsort(snr_rf_max)[::-1]
    roi_rf = roi_rf_sorted[:snr_ranking_num]
    n_sample_rf = rf['data_sorted'].shape[0]
    rf_traces_examples = np.zeros((n_sample_rf,snr_ranking_num))
    for i in range(snr_ranking_num):
        rf_traces_examples[:,i] = rf['data_sorted'][:,snr_rf_max_idx[roi_rf[i]],-1,roi_rf[i]]
    if plot_figure:
        plt.imshow(rf_traces_examples.T,cmap=cmap,norm=norm)
        plt.show()
    return snr_rf_max_idx,roi_rf,n_sample_rf,rf_traces_examples


def rf_to_2d(rf_1d_idx,n_col=6,n_row=4):
    idx_col = rf_1d_idx//n_row
    idx_row = np.mod(rf_1d_idx, n_row)
    return np.array([idx_row,idx_col])

def rf_to_1d(rf_2d_idx,n_col=6,n_row=4):  
    return rf_2d_idx[1]*n_row + rf_2d_idx[0]

def diff_ori(a,b):
    a1 = np.mod(a,180)
    b1 = np.mod(b,180)
    _min = min(a1,b1)
    _max = max(a1,b1)
    c1 = _max - _min
    if c1>90:
        c1 = _min + 180 - _max
    return c1

def diff_dir(a,b):
    a1 = np.mod(a,360)
    b1 = np.mod(b,360)
    _min = min(a1,b1)
    _max = max(a1,b1)
    c1 = _max - _min
    if c1>180:
        c1 = _min + 360 - _max
    return c1

def distance_resp(distance, resp_1, resp_2, resp_bg_1, resp_bg_2, idx_1, idx_2,
                  sub_title=None):
    '''
    distance: distance to RF center, shape (4, 6, n_rois)
    resp_*: response amplitude, shape (4, 6, n_rois)
    resp_bg_*: background response amplitude, shape (n_rois)
    '''
    x_max = 50
    ncols = 4
    labelsize = 20
    fig, axis = plt.subplots(nrows=1, ncols=ncols, figsize=(ncols*5, 5))

    ax_plot_distanceRFcenter_v2(axis[0], distance, resp_1, 50, idx_1, 'red')
    _data = resp_bg_1[idx_1]
    axis[0].bar(x=55, height=_data.mean(), yerr=scipy.stats.sem(_data),
        capsize=5, color='lightgrey', edgecolor='black', width=5)

    ax_plot_distanceRFcenter_v2(axis[1], distance, resp_2, 50, idx_1, 'red')
    _data = resp_bg_2[idx_1]
    axis[1].bar(x=55, height=_data.mean(), yerr=scipy.stats.sem(_data),
        capsize=5, color='lightgrey', edgecolor='black', width=5)

    ax_plot_distanceRFcenter_v2(axis[2], distance, resp_1, 50, idx_2, 'blue')
    _data = resp_bg_1[idx_2]
    axis[2].bar(x=55, height=_data.mean(), yerr=scipy.stats.sem(_data),
        capsize=5, color='lightgrey', edgecolor='black', width=5)

    ax_plot_distanceRFcenter_v2(axis[3], distance, resp_2, 50, idx_2, 'blue')
    _data = resp_bg_2[idx_2]
    axis[3].bar(x=55, height=_data.mean(), yerr=scipy.stats.sem(_data),
        capsize=5, color='lightgrey', edgecolor='black', width=5)

    for i in range(ncols):
        y_min = 0
        y_max = 0
        for c in range(ncols):
            y = axis[c].get_ylim()
            y_min = np.min((y[0], y_min))
            y_max = np.max((y[1], y_max))
        for c in range(ncols):
            axis[c].set_xlim(0, right=x_max+9)
            axis[c].tick_params(axis='both', which='major', labelsize=labelsize)
            axis[c].set_ylim(y_min, y_max)
            axis[c].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.subplots_adjust(wspace=0.3)
    plt.show()

def cal_si(x,y):
    '''
    

    Parameters
    ----------
    x : TYPE
        salient stimulus.
    y : TYPE
        non-salient stimulus.

    Returns
    -------
    TYPE
        saliency index.

    '''
    return (x-y)/(x+y)

# this analysis_sg1_example_ver_0 calculate background based on sg1
def analysis_sg1_example_ver_0(sg1,rf,mb,roi_example,visual_pos,idx_rf_pos, n_non_salient=3,plot_figure=False):
    # calculate the averaged responses when the salient stimulus is at 5 farthest locations from the RF
    n_non_salient = n_non_salient
    dist_salient = np.zeros((rf['n_rows'],rf['n_cols']))
    rf_pos_example = rf['rf_pos_deg_cal'][:,roi_example]
    for i in range(rf['n_rows']):
        for j in range(rf['n_cols']):
            dist_salient[i,j] = scipy.linalg.norm(visual_pos[i,j,:]-rf_pos_example)
    dist_salient_1d = dist_salient.T.ravel()
    dist_salient_1d_idx = np.argsort(dist_salient_1d)
    dist_salient_idx = rf_to_2d(dist_salient_1d_idx)
    
    # analyze salience gratings
    sg1_0 = sg1['sg1_0']
    sg1_90 = sg1['sg1_90']
    if plot_figure:
        plot_rfs_example((sg1_0,sg1_90),roi_example)
    resp_salient_90 = sg1_0[idx_rf_pos[0],idx_rf_pos[1],roi_example]
    resp_salient_0 = sg1_90[idx_rf_pos[0],idx_rf_pos[1],roi_example]

    resp_bk_0_arr = np.zeros(n_non_salient) 
    resp_bk_90_arr = np.zeros(n_non_salient) 
    for i in range(n_non_salient):
        resp_bk_0_arr[i] = sg1_0[dist_salient_idx[0,-1-i],dist_salient_idx[1,-1-i],roi_example]
        resp_bk_90_arr[i] = sg1_90[dist_salient_idx[0,-1-i],dist_salient_idx[1,-1-i],roi_example]
    resp_bk_0 = resp_bk_0_arr.mean()
    resp_bk_90 = resp_bk_90_arr.mean()
    
    if plot_figure:    
        ylim_max = np.max(sg1['data_sorted'][:,:,:-1,roi_example])
        ylim_min = np.min(sg1['data_sorted'][:,:,:-1,roi_example])
        # plot the trace for salient stimulus at 0 deg background
        plot_single_trace(sg1['data_sorted'][:,rf_to_1d(idx_rf_pos),:-1,roi_example],ylim=[ylim_min,ylim_max],title='salient_bk_0')
        # plot the trace for salient stimulus at 90 deg background
        plot_single_trace(sg1['data_sorted'][:,rf_to_1d(idx_rf_pos)+int(sg1['n_stim']/2),:-1,roi_example],ylim=[ylim_min,ylim_max],title='salient_bk_90')
        
        # plot the trace for background stimulus at 0 deg background
        plot_single_trace(sg1['data_sorted'][:,rf_to_1d(dist_salient_idx[:,-1]),:-1,roi_example],ylim=[ylim_min,ylim_max],title='bk_0')
        # plot the trace for background stimulus at 90 deg background
        plot_single_trace(sg1['data_sorted'][:,rf_to_1d(dist_salient_idx[:,-1])+int(sg1['n_stim']/2),:-1,roi_example],ylim=[ylim_min,ylim_max],title='bk_90')
    
    # caculate si for each orientation
    si_0 = cal_si(resp_salient_0,resp_bk_0)
    si_90 = cal_si(resp_salient_90,resp_bk_90)
    # caculate si for each background
    if diff_ori(mb['pref_ori'][roi_example],0)<diff_ori(mb['pref_ori'][roi_example],90):
        osi_bk = cal_si(resp_bk_0,resp_bk_90) #
        osi_salient = cal_si(resp_salient_0,resp_salient_90)
    else:
        osi_bk = cal_si(resp_bk_90,resp_bk_0) #
        osi_salient = cal_si(resp_salient_90,resp_salient_0)
    
    if resp_bk_0>resp_bk_90:
        si_pref = si_0
        si_orth = si_90
    else:
        si_pref = si_90
        si_orth = si_0
    
    sg1_dict = {'si_0':si_0,
                'si_90':si_90,
                'osi_bk':osi_bk,
                'osi_salient':osi_salient,
                'resp_salient_0':resp_salient_0,
                'resp_salient_90':resp_salient_90,
                'resp_bk_0':resp_bk_0,
                'resp_bk_90':resp_bk_90,
                'si_pref':si_pref,
                'si_orth':si_orth}
    return sg1_dict

def analysis_sg1_example(sg1,sg2, rf,mb,roi_example,visual_pos,idx_rf_pos, n_non_salient=5,plot_figure=False):
    
    # analyze salience gratings
    n_cols = sg1['n_cols']
    n_rows = sg1['n_rows']
    # peak_amp = sg1['peak_amp']
    # sg1_0 = np.zeros((n_rows, n_cols, n_rois))
    # sg1_90 = np.zeros((n_rows, n_cols, n_rois))
    n_stim = n_rows*n_cols
    
    #  the following is to recalculate the baseline for each case, which is not used anymore
    # sg1_data = sg1['data_sorted'][:,:,-1,roi_example]
    # baseline = np.zeros(n_stim*2)
    # for i in range(n_stim*2):
    #     baseline[i] = np.sort(sg1_data[:,i])[:3].mean()
            
    # peak_amp = find_peak(sg1_data,baseline=baseline)
    # sg1_0 = peak_amp[0:n_stim].reshape((n_cols, n_rows)).T # for on stimulus
    # sg1_90 = peak_amp[n_stim:n_stim*2].reshape((n_cols, n_rows)).T # for off stimulus   
    
    sg1_0 = sg1['sg1_0'][:,:,roi_example]
    sg1_90 = sg1['sg1_90'][:,:,roi_example]
    if plot_figure:
        plot_rfs_example((sg1_0,sg1_90),roi_example)
    
    
    resp_salient_90 = sg1_0[idx_rf_pos[0],idx_rf_pos[1]]
    resp_salient_0 = sg1_90[idx_rf_pos[0],idx_rf_pos[1]]

    sg2['n_background'] = 2
    n_rows = sg2['n_background']
    n_cols = int(sg2['n_stim']/sg2['n_background'])
    # sg2_data = sg2['data_sorted'][:,:,-1,roi_example]
    # sg2_baseline = np.zeros(sg2_data.shape[1])
    # for i in range(sg2_data.shape[1]):
    #     sg2_baseline[i] = np.sort(sg2_data[:,i])[:3].mean()
    # sg2_peak_amp = find_peak(sg2_data,baseline=sg2_baseline)
    # sg2_0_90 = sg2_peak_amp.reshape((n_rows,n_cols))

    sg2_0_90 = sg2['sg2_0_90'][:,:,roi_example]
    resp_bk_0 = sg2_0_90[0,0]
    resp_bk_90 = sg2_0_90[1,0]
    
    # caculate si for each orientation
    si_0 = cal_si(resp_salient_0,resp_bk_0)
    si_90 = cal_si(resp_salient_90,resp_bk_90)
    # caculate si for each background
    if diff_ori(mb['pref_ori'][roi_example],0)<diff_ori(mb['pref_ori'][roi_example],90):
        osi_bk = cal_si(resp_bk_0,resp_bk_90) #
        osi_salient = cal_si(resp_salient_0,resp_salient_90)
    else:
        osi_bk = cal_si(resp_bk_90,resp_bk_0) #
        osi_salient = cal_si(resp_salient_90,resp_salient_0)
    
    if resp_bk_0>resp_bk_90:
        si_pref = si_0
        si_orth = si_90
    else:
        si_pref = si_90
        si_orth = si_0
    
    sg1_dict = {'si_0':si_0,
                'si_90':si_90,
                'osi_bk':osi_bk,
                'osi_salient':osi_salient,
                'resp_salient_0':resp_salient_0,
                'resp_salient_90':resp_salient_90,
                'resp_bk_0':resp_bk_0,
                'resp_bk_90':resp_bk_90,
                'si_pref':si_pref,
                'si_orth':si_orth}
    return sg1_dict

def analysis_smd1_example_ver_0(smd1,rf,mb,roi_example,visual_pos, idx_rf_pos,n_non_salient=3,plot_figure=False):
    n_non_salient = n_non_salient
    dist_salient = np.zeros((rf['n_rows'],rf['n_cols']))
    rf_pos_example = rf['rf_pos_deg_cal'][:,roi_example]
    for i in range(rf['n_rows']):
        for j in range(rf['n_cols']):
            dist_salient[i,j] = scipy.linalg.norm(visual_pos[i,j,:]-rf_pos_example)
    dist_salient_1d = dist_salient.T.ravel()
    dist_salient_1d_idx = np.argsort(dist_salient_1d)
    dist_salient_idx = rf_to_2d(dist_salient_1d_idx)
    
    smd1_0 = smd1['smd1_0']
    smd1_90 = smd1['smd1_90']
    smd1_180 = smd1['smd1_180']
    smd1_270 = smd1['smd1_270']
    if plot_figure: 
        plot_rfs_example((smd1_0,smd1_90,smd1_180,smd1_270),roi_example)
    resp_salient_180 = smd1_0[idx_rf_pos[0],idx_rf_pos[1],roi_example]
    resp_salient_270 = smd1_90[idx_rf_pos[0],idx_rf_pos[1],roi_example]
    resp_salient_0 = smd1_180[idx_rf_pos[0],idx_rf_pos[1],roi_example]
    resp_salient_90 = smd1_270[idx_rf_pos[0],idx_rf_pos[1],roi_example]

    resp_bk_0_arr = np.zeros(n_non_salient) 
    resp_bk_90_arr = np.zeros(n_non_salient) 
    resp_bk_180_arr = np.zeros(n_non_salient) 
    resp_bk_270_arr = np.zeros(n_non_salient) 
    for i in range(n_non_salient):
        resp_bk_0_arr[i] = smd1_0[dist_salient_idx[0,-1-i],dist_salient_idx[1,-1-i],roi_example]
        resp_bk_90_arr[i] = smd1_90[dist_salient_idx[0,-1-i],dist_salient_idx[1,-1-i],roi_example]
        resp_bk_180_arr[i] = smd1_180[dist_salient_idx[0,-1-i],dist_salient_idx[1,-1-i],roi_example]
        resp_bk_270_arr[i] = smd1_270[dist_salient_idx[0,-1-i],dist_salient_idx[1,-1-i],roi_example]
    resp_bk_0 = resp_bk_0_arr.mean()
    resp_bk_90 = resp_bk_90_arr.mean()
    resp_bk_180 = resp_bk_180_arr.mean()
    resp_bk_270 = resp_bk_270_arr.mean()
    
    if plot_figure:     
        ylim_max = np.max(smd1['data_sorted'][:,:,:-1,roi_example])
        ylim_min = np.min(smd1['data_sorted'][:,:,:-1,roi_example])
        # plot the trace for salient stimulus at 0 deg background
        plot_single_trace(smd1['data_sorted'][:,rf_to_1d(idx_rf_pos),:-1,roi_example],ylim=[ylim_min,ylim_max],title='salient_smd1_bk_0')
        # plot the trace for salient stimulus at 90 deg background
        plot_single_trace(smd1['data_sorted'][:,rf_to_1d(idx_rf_pos)+int(smd1['n_stim']/4),:-1,roi_example],ylim=[ylim_min,ylim_max],title='salient_smd1_bk_90')
        plot_single_trace(smd1['data_sorted'][:,rf_to_1d(idx_rf_pos)+int(smd1['n_stim']/4)*2,:-1,roi_example],ylim=[ylim_min,ylim_max],title='salient_smd1_bk_180')
        plot_single_trace(smd1['data_sorted'][:,rf_to_1d(idx_rf_pos)+int(smd1['n_stim']/4)*3,:-1,roi_example],ylim=[ylim_min,ylim_max],title='salient_smd1_bk_270')
        
        # plot the trace for background stimulus at 0 deg background
        plot_single_trace(smd1['data_sorted'][:,rf_to_1d(dist_salient_idx[:,-1]),:-1,roi_example],ylim=[ylim_min,ylim_max],title='bk_0')
        # plot the trace for background stimulus at 90 deg background
        plot_single_trace(smd1['data_sorted'][:,rf_to_1d(dist_salient_idx[:,-1])+int(smd1['n_stim']/4),:-1,roi_example],ylim=[ylim_min,ylim_max],title='bk_90')
        plot_single_trace(smd1['data_sorted'][:,rf_to_1d(dist_salient_idx[:,-1])+int(smd1['n_stim']/4)*2,:-1,roi_example],ylim=[ylim_min,ylim_max],title='bk_180')
        plot_single_trace(smd1['data_sorted'][:,rf_to_1d(dist_salient_idx[:,-1])+int(smd1['n_stim']/4)*3,:-1,roi_example],ylim=[ylim_min,ylim_max],title='bk_270')
    
    # caculate si for each orientation
    si_0 = cal_si(resp_salient_0,resp_bk_0)
    si_90 = cal_si(resp_salient_90,resp_bk_90)
    si_180 = cal_si(resp_salient_180,resp_bk_180)
    si_270 = cal_si(resp_salient_270,resp_bk_270)
    # caculate si for each background
    _diff_dir_min = 360
    theta_pref = 0
    for theta in [0,90,180,270]:
        if diff_dir(mb['pref_dir'][roi_example],theta)<_diff_dir_min:
            _diff_dir_min = diff_dir(mb['pref_dir'][roi_example],theta)
            theta_pref = theta
    theta_null = np.mod(theta_pref+180,360)
    dsi_bk = eval('cal_si(resp_bk_{},resp_bk_{})'.format(theta_pref,theta_null)) #
    dsi_salient = eval('cal_si(resp_salient_{},resp_salient_{})'.format(theta_pref,theta_null)) #
    
    si_arr = np.array([si_0,si_90,si_180,si_270])
    si_max_idx = np.argmax(si_arr)
    si_pref = si_arr[si_max_idx]
    si_null = si_arr[np.mod(si_max_idx+2,4)]
    si_pref_cw_90 = si_arr[np.mod(si_max_idx+1,4)]
    si_pref_ccw_90 = si_arr[np.mod(si_max_idx+3,4)]
    smd1_dict = {'si_0':si_0,
                'si_90':si_90,
                'si_180':si_180,
                'si_270':si_270,
                'dsi_bk':dsi_bk,
                'dsi_salient':dsi_salient,
                'resp_salient_0':resp_salient_0,
                'resp_salient_90':resp_salient_90,
                'resp_salient_180':resp_salient_180,
                'resp_salient_270':resp_salient_270,
                'resp_bk_0':resp_bk_0,
                'resp_bk_90':resp_bk_90,
                'resp_bk_180':resp_bk_180,
                'resp_bk_270':resp_bk_270,
                'si_pref':si_pref,
                'si_null':si_null,
                'si_pref_cw_90':si_pref_cw_90,
                'si_pref_ccw_90':si_pref_ccw_90
                }
    return smd1_dict

def analysis_smd1_example(smd1,smd2, rf,mb,roi_example,visual_pos, idx_rf_pos,n_non_salient=5,plot_figure=False):
    # analyze salience gratings
    n_cols = smd1['n_cols']
    n_rows = smd1['n_rows']
    # peak_amp = sg1['peak_amp']
    # sg1_0 = np.zeros((n_rows, n_cols, n_rois))
    # sg1_90 = np.zeros((n_rows, n_cols, n_rois))
    n_stim = n_rows*n_cols
    
    # smd1_data = smd1['data_sorted'][:,:,-1,roi_example]
    # baseline = np.zeros(smd1_data.shape[1])
    # for i in range(smd1_data.shape[1]):
    #     baseline[i] = np.sort(smd1_data[:,i])[:3].mean()
    # peak_amp = find_peak(smd1_data,baseline=baseline)
    # smd1_0 = peak_amp[0:n_stim].reshape((n_cols, n_rows)).T # for on stimulus
    # smd1_90 = peak_amp[n_stim:n_stim*2].reshape((n_cols, n_rows)).T # for off stimulus   
    # smd1_180 = peak_amp[n_stim*2:n_stim*3].reshape((n_cols, n_rows)).T # for on stimulus
    # smd1_270 = peak_amp[n_stim*3:n_stim*4].reshape((n_cols, n_rows)).T # for off stimulus   
    
    
    smd1_0 = smd1['smd1_0'][:,:,roi_example]
    smd1_90 = smd1['smd1_90'][:,:,roi_example]
    smd1_180 = smd1['smd1_180'][:,:,roi_example]
    smd1_270 = smd1['smd1_270'][:,:,roi_example]
    
    if plot_figure: 
        plot_rfs_example((smd1_0,smd1_90,smd1_180,smd1_270),roi_example)
    resp_salient_180 = smd1_0[idx_rf_pos[0],idx_rf_pos[1]]
    resp_salient_270 = smd1_90[idx_rf_pos[0],idx_rf_pos[1]]
    resp_salient_0 = smd1_180[idx_rf_pos[0],idx_rf_pos[1]]
    resp_salient_90 = smd1_270[idx_rf_pos[0],idx_rf_pos[1]]
    

    smd2['n_background'] = 4
    n_rows = smd2['n_background']
    n_cols = int(smd2['n_stim']/smd2['n_background'])
    # smd2_data = smd2['data_sorted'][:,:,-1,roi_example]
    # smd2_baseline = np.zeros(smd2_data.shape[1])
    # for i in range(smd2_data.shape[1]):
    #     smd2_baseline[i] = np.sort(smd2_data[:,i])[:3].mean()
    # smd2_peak_amp = find_peak(smd2_data,baseline=smd2_baseline)
    # smd2_2d = smd2_peak_amp.reshape((n_rows,n_cols))
    
    smd2_2d = smd2['smd2'][:,:,roi_example]

    resp_bk_0 = smd2_2d[0,0]
    resp_bk_90 = smd2_2d[1,3]
    resp_bk_180 = smd2_2d[2,6]
    resp_bk_270 = smd2_2d[3,9]
    
    
    # caculate si for each orientation
    si_0 = cal_si(resp_salient_0,resp_bk_0)
    si_90 = cal_si(resp_salient_90,resp_bk_90)
    si_180 = cal_si(resp_salient_180,resp_bk_180)
    si_270 = cal_si(resp_salient_270,resp_bk_270)
    # caculate si for each background
    _diff_dir_min = 360
    theta_pref = 0
    for theta in [0,90,180,270]:
        if diff_dir(mb['pref_dir'][roi_example],theta)<_diff_dir_min:
            _diff_dir_min = diff_dir(mb['pref_dir'][roi_example],theta)
            theta_pref = theta
    theta_null = np.mod(theta_pref+180,360)
    dsi_bk = eval('cal_si(resp_bk_{},resp_bk_{})'.format(theta_pref,theta_null)) #
    dsi_salient = eval('cal_si(resp_salient_{},resp_salient_{})'.format(theta_pref,theta_null)) #
    
    si_arr = np.array([si_0,si_90,si_180,si_270])
    si_max_idx = np.argmax(si_arr)
    si_pref = si_arr[si_max_idx]
    si_null = si_arr[np.mod(si_max_idx+2,4)]
    si_pref_cw_90 = si_arr[np.mod(si_max_idx+1,4)]
    si_pref_ccw_90 = si_arr[np.mod(si_max_idx+3,4)]
    smd1_dict = {'si_0':si_0,
                'si_90':si_90,
                'si_180':si_180,
                'si_270':si_270,
                'dsi_bk':dsi_bk,
                'dsi_salient':dsi_salient,
                'resp_salient_0':resp_salient_0,
                'resp_salient_90':resp_salient_90,
                'resp_salient_180':resp_salient_180,
                'resp_salient_270':resp_salient_270,
                'resp_bk_0':resp_bk_0,
                'resp_bk_90':resp_bk_90,
                'resp_bk_180':resp_bk_180,
                'resp_bk_270':resp_bk_270,
                'si_pref':si_pref,
                'si_null':si_null,
                'si_pref_cw_90':si_pref_cw_90,
                'si_pref_ccw_90':si_pref_ccw_90
                }
    return smd1_dict

def analysis_smg1_example(smg1, rf, roi_example, visual_pos, idx_rf_pos, n_non_salient=3):
    # this analysis_smg1_example calculate background based on smg1
    # calculate the averaged responses when the salient stimulus is at n_non_salient farthest locations from the RF
    n_non_salient = n_non_salient
    dist_salient = np.zeros((rf['n_rows'], rf['n_cols']))
    rf_pos_example = rf['rf_pos_deg_cal'][:,roi_example]
    for i in range(rf['n_rows']):
        for j in range(rf['n_cols']):
            dist_salient[i, j] = scipy.linalg.norm(visual_pos[i,j,:] - rf_pos_example)
    dist_salient_1d = dist_salient.T.ravel()
    dist_salient_1d_idx = np.argsort(dist_salient_1d)
    dist_salient_idx = rf_to_2d(dist_salient_1d_idx)
    
    # analyze salience gratings
    smg1_0 = smg1['smg1_0']
    smg1_90 = smg1['smg1_90']
    smg1_180 = smg1['smg1_180']
    smg1_270 = smg1['smg1_270']

    resp_salient_0 = smg1_180[idx_rf_pos[0], idx_rf_pos[1], roi_example]
    resp_salient_90 = smg1_270[idx_rf_pos[0], idx_rf_pos[1], roi_example]
    resp_salient_180 = smg1_0[idx_rf_pos[0], idx_rf_pos[1], roi_example]
    resp_salient_270 = smg1_90[idx_rf_pos[0], idx_rf_pos[1], roi_example]

    resp_bk_0_arr = np.zeros(n_non_salient)
    resp_bk_90_arr = np.zeros(n_non_salient)
    resp_bk_180_arr = np.zeros(n_non_salient)
    resp_bk_270_arr = np.zeros(n_non_salient)
    for i in range(n_non_salient):
        resp_bk_0_arr[i] = smg1_0[dist_salient_idx[0,-1-i], dist_salient_idx[1,-1-i], roi_example]
        resp_bk_90_arr[i] = smg1_90[dist_salient_idx[0,-1-i], dist_salient_idx[1,-1-i], roi_example]
        resp_bk_180_arr[i] = smg1_180[dist_salient_idx[0,-1-i], dist_salient_idx[1,-1-i], roi_example]
        resp_bk_270_arr[i] = smg1_270[dist_salient_idx[0,-1-i], dist_salient_idx[1,-1-i], roi_example]
    resp_bk_0 = resp_bk_0_arr.mean()
    resp_bk_90 = resp_bk_90_arr.mean()
    resp_bk_180 = resp_bk_180_arr.mean()
    resp_bk_270 = resp_bk_270_arr.mean()
    
    smg1_dict = {'resp_salient_0': resp_salient_0,
                'resp_salient_90': resp_salient_90,
                'resp_salient_180': resp_salient_180,
                'resp_salient_270': resp_salient_270,
                'resp_bk_0': resp_bk_0,
                'resp_bk_90': resp_bk_90,
                'resp_bk_180': resp_bk_180,
                'resp_bk_270': resp_bk_270}
    return smg1_dict

# the three functions below is defined for boolen array, which work but are not easy to understand
# def jaccard(y_true, y_pred):
#     """Calculate Jaccard similarity coefficient

#     Ranges from 0 to 1. Higher values indicate greater similarity.

#     Parameters
#     ----------
#     y_true : list or array
#         Boolean indication of cluster membership (1 if belonging, 0 if
#         not belonging) for actual labels
#     y_pred : list or array
#         Boolean indication of cluster membership (1 if belonging, 0 if
#         not belonging) for predicted labels

#     Returns
#     -------
#     float
#         Jaccard coefficient value
#     """
#     if type(y_true) is list:
#         y_true = np.array(y_true)

#     if type(y_pred) is list:
#         y_pred = np.array(y_pred)
#     # return float(np.sum(y_true & y_pred)) / (np.sum(y_true) + np.sum(y_pred) - np.sum(y_true & y_pred))
#     return float(np.sum(np.logical_and(y_true,y_pred))) / float(np.sum(np.logical_or(y_true,y_pred)))


# def jsc(x_list):
#     n = len(x_list)
#     sum_x = np.zeros(x_list[0].size)
#     for x in x_list:
#         sum_x = sum_x+x
#     return float(np.sum(sum_x==n))/float(np.sum(sum_x>0))


# def jsc_permute(idx_sg1_list,n_random=1000):
#     k=0
#     jsc_sg1_permute = np.zeros(n_random)
#     for i in range(n_random):
#         _idx_sg1_list=[]
#         for j in range(len(idx_sg1_list)):
#             _idx_sg1 = np.ones_like(idx_sg1_list[j])*False
#             _n = idx_sg1_list[j].size
#             _m = int(np.sum(idx_sg1_list[j]))
#             np.random.seed(k)
#             _idx_sg1[np.random.choice(range(_n),_m,replace=False)] = True
#             _idx_sg1_list.append(_idx_sg1)
#             k+=1
#         jsc_sg1_permute[i] = jsc(_idx_sg1_list) 
#     jsc_sg1_permute_mean = np.mean(jsc_sg1_permute)
#     jsc_sg1_permute_std = np.std(jsc_sg1_permute)
    
#     return jsc_sg1_permute_mean,jsc_sg1_permute_std

def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None):
    """ 
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    plt.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs)


def jsc(x_list):
    n = len(x_list)
    intersect_arr = x_list[0]
    union_arr = x_list[0]
    for i in range(1,n):
        intersect_arr = np.intersect1d(intersect_arr,x_list[i])
        union_arr = np.union1d(union_arr,x_list[i])
    return intersect_arr.size/union_arr.size


def jsc_permute(x_list,n_random=1000):
    jsc_x = jsc(x_list)
    print(jsc_x)
    n = len(x_list)
    size_x = np.zeros(n,dtype=int)
    for i in range(n):
        size_x[i] = x_list[i].size
        
    union_arr = x_list[0]
    for i in range(1,n):
        union_arr = np.union1d(union_arr,x_list[i])
    size_union = union_arr.size
    
    k=0
    jsc_x_permute = np.zeros(n_random)
    for i in range(n_random):
        _x_list=[]
        for j in range(n):
            _idx_x = np.ones(size_union)<0
            np.random.seed(k)
            _idx_x[np.random.choice(range(size_union),size_x[j],replace=False)] = True
            _x_list.append(union_arr[_idx_x])
            k+=1
        jsc_x_permute[i] = jsc(_x_list) 
    jsc_x_permute_mean = np.mean(jsc_x_permute)
    jsc_x_permute_std = np.std(jsc_x_permute)
    # pvalue = scipy.stats.ttest_1samp(jsc_x_permute,popmean=jsc_x)
    count_larger = np.sum(jsc_x_permute-jsc_x>0)
    count_smaller = n_random-count_larger
    pvalue = np.min([count_larger,count_smaller])/n_random
    return jsc_x_permute,pvalue

def find_outliers(data,thr=3):
    mean = np.mean(data)
    std = np.std(data)
    z_score = (data - mean) / std
    idx_outlier = np.where(abs(z_score)>thr)[0]
    idx_normal = np.where(abs(z_score)<=thr)[0]
    return idx_outlier,idx_normal

def rf_consistence(x,y,rf_dist_thr):
    n_rois = np.shape(x)[1]
    rf_good_roi = np.ones(n_rois,dtype=int)*False
    idx_not_nan_x = np.logical_not(np.logical_and(np.isnan(x[0,:]),x[1,:]))
    idx_not_nan_y = np.logical_not(np.logical_and(np.isnan(y[0,:]),y[1,:]))
    idx_not_nan = np.logical_and(idx_not_nan_x,idx_not_nan_y)
    rf_dist_arr = np.ones(n_rois)*np.nan
    rf_dist_arr[idx_not_nan] = scipy.linalg.norm(x[:,idx_not_nan]-y[:,idx_not_nan],axis=0)
    rf_good_roi[idx_not_nan] = rf_dist_arr[idx_not_nan] <rf_dist_thr
    return rf_good_roi.astype(bool)

def pix_to_visual_angle(loc_pix,para):
    monitor_pix = np.array([para['width_pix'],para['height_pix']])
    monitor_center_pix = monitor_pix/2
    loc_pix_centered = loc_pix - monitor_center_pix
    loc_pix_centered[1] = -loc_pix_centered[1] #inverse y axis
    monitor_mm = np.array([para['width'],para['height']])
    loc_mm_centered = loc_pix_centered/monitor_pix.mean()*monitor_mm.mean()
    loc_deg_centered = np.rad2deg(np.arctan(loc_mm_centered/para['distance']))
    return para['monitor_center'] + loc_deg_centered

def hist_customize(data, bins=20, color='blue', label=False, vline=None, xlim=None, xticks=True, xlocals=None, exclude_extremum=False):
    '''
    xlocals: the locations of labels of x axis, is a list
    vline: the vertical line to mark the value, x axis location
    '''
    good = np.logical_not(np.isnan(data))
    if exclude_extremum:
        not_one = abs(data) < 0.95 # to exclude the values around 1
        good = np.logical_and(good, not_one)
    
    # print('number of samples: {};'.format(good.sum()))
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fontsize = 20

    facecolor = mcolors.to_rgb(color)
    facecolor_alpha = (facecolor[0], facecolor[1], facecolor[2], 0.5)
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
        ax.hist(data[good], bins=bins, range=xlim, facecolor=facecolor_alpha, edgecolor='black')
    else:
        ax.hist(data[good], bins=bins, facecolor=facecolor_alpha, edgecolor='black')

    # if xlim is not None:
    #     ax.set_xlim(left=xlim[0], right=xlim[1])
    #     ax.hist(data[good], bins=bins, range=xlim, color=color, edgecolor='black')
    # else:
    #     ax.hist(data[good], bins=bins, color=color, edgecolor='black')

    if label:
        # ax.axvline(-0.3, color='magenta', linestyle='--', linewidth=3)
        ax.axvline(0, color='cyan', linestyle='--', linewidth=2) # lime, limegreen
    if vline is not None:
        for v in vline:
            ax.axvline(v, color='black', linestyle='--', linewidth=2)
    if not xticks:
        ax.set_xticklabels([])
    if xlocals is not None:
        plt.xticks(xlocals)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.spines[['top', 'right']].set_visible(False)
    # print('mean ± std: {:.2f} ± {:.2f};'.format(np.mean(data[good]), np.std(data[good])))
    plt.show()

def hist_customize_stacked(data_bottom, data_up, xlim, bins=20, label=False, xticks=True, xlocals=None, exclude_extremum=False, title=None):
    '''
    xlocals: the locations of labels of x axis, is a list
    '''
    good = np.logical_not(np.isnan(data_bottom))
    if exclude_extremum:
        not_one = np.logical_and(abs(data_bottom) < 0.95, abs(data_up) < 0.95) # to exclude the values around 1
        good = np.logical_and(good, not_one)
    
    print('number of samples: {};'.format(good.sum()))
    width = 4
    height = 2.5
    fig, ax = plt.subplots(figsize=(width, height))
    fontsize = 20

    hist_bottom, edges = np.histogram(data_bottom[good], bins=bins, range=xlim)
    hist_up, edges = np.histogram(data_up, bins=bins, range=xlim)
    width = edges[1]-edges[0]
    ax.bar(edges[:-1], hist_bottom, width=width, color='green', edgecolor='black', align='edge')
    ax.bar(edges[:-1], hist_up, bottom=hist_bottom, width=width, color='orange', edgecolor='black', align='edge')
    ax.set_xlim(left=xlim[0], right=xlim[1])
        
    if label:
        # ax.axvline(-0.3, color='magenta', linestyle='--', linewidth=3)
        ax.axvline(0, color='cyan', linestyle='--', linewidth=3) # lime, limegreen
    if not xticks:
        ax.set_xticklabels([])
    if xlocals is not None:
        plt.xticks(xlocals)
    ax.tick_params(axis='both', labelsize=fontsize)
    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.show()

def hist2d_hist(x, y, title=None):
    '''
    Creates a 2D histogram scatter plot with additional histograms on the side and a colorbar at the top
    '''
    # Start with a square Figure.
    fig = plt.figure(figsize=(6, 6))

    # Add a gridspec with two rows and two columns and a ratio of 1 to 4
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 20),
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05) #
    
        # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_colorbar = fig.add_subplot(gs[0, 0])

    # the scatter plot:
    _, _, _, image = ax.hist2d(x, y, bins=20, cmap='Greys')
    # ax.set_xlabel('Distance to RF center')
    # ax.set_ylabel('SI')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # Plot polynomial fit
    coeffs = np.polyfit(x, y, deg=1)
    poly = np.poly1d(coeffs)
    x_fit = (np.min(x)+5, np.max(x)-5)
    print(x_fit)
    y_fit = poly(x_fit)
    ax.plot(x_fit, y_fit, color='blue', label='Polyfit', linestyle='--')

    # Plot Pearson correlation coefficient
    corr_coef, _ = scipy.stats.pearsonr(x, y)
    ax.text(0.05, 0.95, 'r = {:.2f}'.format(corr_coef), transform=ax.transAxes)
    ax.legend(loc='upper right')

    # no labels
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.hist(y, orientation='horizontal', color='gray')
    y_mean = np.mean(y)
    ax_histy.axhline(y_mean, color='red', label='Mean')
    # ax_histy.legend(loc='upper right')
    # ax_histy.spines['top'].set_visible(False)
    # ax_histy.spines['right'].set_visible(False)

    fig.colorbar(image, cax=ax_colorbar, location='bottom', orientation='horizontal')
    
    fig.tight_layout()
    plt.suptitle(title)

def hist2d_hist_v2(x, y, cmap='Greys', label_r=True, label_line=0.3, mean_line=True, identical_line=True, exclude_extremum=True, title=None):
    '''
    Creates a 2D histogram scatter plot with additional histograms on the right and top side and a colorbar
    '''
    not_nan = np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y)))
    not_zero = np.logical_and(x != 0, y != 0)
    good = np.logical_and(not_nan, not_zero)
    if exclude_extremum:
        not_one = np.logical_and(abs(x) < 0.95, abs(y) < 0.95)
        good = np.logical_and(good, not_one)
    print('number of samples: {};'.format(good.sum()))
    x = x[good]
    y = y[good]

    figsize = (5, 5.5)
    fig = plt.figure(figsize=figsize)
    fontsize = np.min(figsize) * 4
    bins = 20
    # Add a gridspec with two rows and two columns and a ratio of 1 to 4
    # Also adjust the subplot parameters for a square plot.
    gs = fig.add_gridspec(nrows=4, ncols=2,  width_ratios=(4, 1), height_ratios=(1, 4, 0.3, 0.2), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
    
        # Create the Axes.
    ax = fig.add_subplot(gs[1, 0])
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_histx = fig.add_subplot(gs[0, 0])
    ax_colorbar = fig.add_subplot(gs[3, 0])

    # the scatter plot:
    _, _, _, image = ax.hist2d(x, y, bins=bins, cmap=cmap)
    # ax.set_aspect('equal')
    # ax.set_xlabel('Distance to RF center')
    # ax.set_ylabel('SI')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    left, right = ax.get_xlim()
    bottom, top = ax.get_ylim()

    # Plot polynomial fit
    coeffs = np.polyfit(x, y, deg=1)
    poly = np.poly1d(coeffs)
    x_fit = (np.min(x)+5, np.max(x)-5)
    y_fit = poly(x_fit)
    
    _, pvalue = scipy.stats.ttest_rel(x, y)
    # Plot Pearson correlation coefficient
    corr_coef, _ = scipy.stats.pearsonr(x, y)
    if label_r:
        ax.plot(x_fit, y_fit, color='blue', label='Polyfit', linestyle='--')
        ax.text(0.05, 0.9, 'r = {:.2f}'.format(corr_coef), transform=ax.transAxes)
        # ax.legend(loc='upper right')

    print('r = {:.2f}, p = {:.2e};'.format(corr_coef, pvalue))
    print('x={:.3f}±{:.2e}, y={:.3f}±{:.2e};'.format(np.mean(x), np.std(x), np.mean(y), np.std(y)))
    if identical_line:
        ax.plot(x_fit, x_fit, color='black', label='Identical', linestyle='--')
    ax.tick_params(axis='both', labelsize=fontsize)

    # no labels
    ax_histy.tick_params(axis="y", labelleft=False)
    ax_histy.hist(y, bins=bins, orientation='horizontal', color='lightgray', edgecolor='black')
    
    # ax_histy.legend(loc='upper right')
    # ax_histy.spines['top'].set_visible(False)
    # ax_histy.spines['right'].set_visible(False)
    ax_histy.set_ylim(bottom, top)
    ax_histy.tick_params(axis='x', labelsize=fontsize)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histx.hist(x, bins=bins, color='lightgray', edgecolor='black')
   
    ax_histx.set_xlim(left, right)
    ax_histx.tick_params(axis='y', labelsize=fontsize)

    if mean_line:
        x_mean = np.mean(x)
        ax_histx.axvline(x_mean, color='yellow', label='Mean', linestyle='--', linewidth=3)
        y_mean = np.mean(y)
        ax_histy.axhline(y_mean, color='yellow', label='Mean', linestyle='--', linewidth=3)
    
    if label_line is not None:
        ax_histx.axvline(-label_line, color='brown', label='Threshold', linestyle='--', linewidth=3)
        ax_histx.axvline(label_line, color='brown', label='Threshold', linestyle='--', linewidth=3)
        ax_histy.axhline(-label_line, color='brown', label='Threshold', linestyle='--', linewidth=3)
        ax_histy.axhline(label_line, color='brown', label='Threshold', linestyle='--', linewidth=3)
    
    cbar = fig.colorbar(image, cax=ax_colorbar, location='bottom', orientation='horizontal')
    cbar.ax.tick_params(labelsize=15)
    fig.tight_layout()
    plt.suptitle(title)

def hist2d_colorbar(x, y, cmap='Greys', xlim=(None, None), label=True, fit_line=True, identical_line=True, exclude_extremum=True, title=None):
    '''
    xlim: the limit of x axis
    '''
    good = np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y)))
    # not_zero = np.logical_and(x != 0, y != 0)
    # good = np.logical_and(not_nan, not_zero)
    if exclude_extremum:
        not_one = abs(y) < 0.95
        good = np.logical_and(good, not_one)
    x = x[good]
    y = y[good]
    print('number of samples: {};'.format(good.sum()))

    figsize = (6, 5)
    # Start with a square Figure.
    fig, ax = plt.subplots(figsize=figsize)
    fontsize = np.min(figsize) * 4

    # the scatter plot:
    _, _, _, image = ax.hist2d(x, y, bins=20, cmap=cmap, norm=matplotlib.colors.LogNorm())
    # ax.set_xlabel('Distance to RF center')
    # ax.set_ylabel('SI')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # Plot polynomial fit
    coeffs = np.polyfit(x, y, deg=1)
    poly = np.poly1d(coeffs)
    x_fit = (np.min(x), np.max(x)+1)
    # print(x_fit)
    y_fit = poly(x_fit)

    if fit_line:
        ax.plot(x_fit, y_fit, color='cyan', label='Polyfit', linestyle='--', linewidth=3) # magenta

    if identical_line:
        ax.plot(x_fit, x_fit, color='black', label='Identical', linestyle='--', linewidth=3)

    _, pvalue = scipy.stats.ttest_rel(x, y)
    # Plot Pearson correlation coefficient
    corr_coef, _ = scipy.stats.pearsonr(x, y)
    if label:
        ax.text(0.05, 0.95, 'r = {:.2f}'.format(corr_coef), transform=ax.transAxes)
        # ax.legend(loc='upper right')
    print('r = {:.2f}, p = {:.2e};'.format(corr_coef, pvalue))
    print('x={:.3f}±{:.2e}, y={:.3f}±{:.2e};'.format(np.mean(x), np.std(x), np.mean(y), np.std(y)))
    # print(scipy.stats.ttest_1samp(x-y, popmean=0))
    
    left, right = ax.get_xlim()
    print('xlim: {}, {};'.format(left, right))
    bottom, top = ax.get_ylim()
    print('ylim: {}, {};'.format(bottom, top))
    locs, labels = plt.xticks()
    # print(locs, labels)
    locs = [0, 10, 20, 30, 40, 50]
    # plt.xticks(locs)
    locs = [10, 20, 30, 40]
    # plt.yticks(locs)

    if xlim[0] is not None:
        ax.set_xlim(left=xlim[0])
        # ax.set_ylim(bottom=xlim[0])

    if xlim[1] is None:
        ax.set_xlim(right=right)
    else:
        ax.set_xlim(right=xlim[1])
    # ax.set_ylim(bottom=-70, top=170)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar = fig.colorbar(image)
    locs, labels = plt.yticks()
    locs = cbar.get_ticks()
    labels = cbar.get_ticks()
    print(locs, labels)
    labels = ['$10^{}$'.format('{' + str(int(np.log10(loc*0.1))) + '}') for loc in locs] # convert the unit from 0.1 s to 1 s
    cbar.set_ticklabels(labels)
    cbar.ax.tick_params(labelsize=15)
    fig.tight_layout()
    plt.suptitle(title)
    plt.show()

def hist2d_colorbar_log(x, y, cmap='Greys', xlim=[None, None], label=True, fit_line=True, identical_line=True, exclude_extremum=True, title=None):
    '''
    xlim: the limit of x axis
    '''
    good = np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y)))
    # not_zero = np.logical_and(x != 0, y != 0)
    # good = np.logical_and(not_nan, not_zero)
    if exclude_extremum:
        not_one = abs(y) < 0.95
        good = np.logical_and(good, not_one)
    x = x[good]
    y = y[good]
    print('number of samples: {};'.format(good.sum()))

    figsize = (6, 5)
    # Start with a square Figure.
    fig, ax = plt.subplots(figsize=figsize)
    fontsize = np.min(figsize) * 4

    # the scatter plot:
    if (np.isnan(xlim)).any():
        _, _, _, image = ax.hist2d(x, y, bins=20, cmap=cmap, norm=matplotlib.colors.LogNorm())
    else:
        ymin = y.min()
        ymax = y.max()
        _, _, _, image = ax.hist2d(x, y, bins=20, cmap=cmap, norm=matplotlib.colors.LogNorm(), range=[xlim, [ymin, ymax]])
    # ax.set_xlabel('Distance to RF center')
    # ax.set_ylabel('SI')
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    # Plot polynomial fit
    coeffs = np.polyfit(x, y, deg=1)
    poly = np.poly1d(coeffs)
    x_fit = (np.min(x), np.max(x)+1)
    # print(x_fit)
    y_fit = poly(x_fit)

    if fit_line:
        ax.plot(x_fit, y_fit, color='cyan', label='Polyfit', linestyle='--', linewidth=3) # magenta

    if identical_line:
        ax.plot(x_fit, x_fit, color='black', label='Identical', linestyle='--', linewidth=3)

    _, pvalue = scipy.stats.ttest_rel(x, y)
    # Plot Pearson correlation coefficient
    corr_coef, _ = scipy.stats.pearsonr(x, y)
    print(corr_coef)
    if label:
        ax.text(0.05, 0.95, 'r = {:.2f}'.format(corr_coef), transform=ax.transAxes)
        # ax.legend(loc='upper right')
    print('r = {:.2f}, p = {:.2e};'.format(corr_coef, pvalue))
    print('x={:.3f}±{:.2e}, y={:.3f}±{:.2e};'.format(np.mean(x), np.std(x), np.mean(y), np.std(y)))
    # print(scipy.stats.ttest_1samp(x-y, popmean=0))
    
    left, right = ax.get_xlim()
    print('xlim: {}, {};'.format(left, right))
    bottom, top = ax.get_ylim()
    print('ylim: {}, {};'.format(bottom, top))
    locs, labels = plt.xticks()
    # print(locs, labels)
    locs = [-0.3, 0, 0.3, 0.6]
    plt.xticks(locs)
    locs = [10, 20, 30, 40]
    # plt.yticks(locs)

    if xlim[0] is not None:
        ax.set_xlim(left=xlim[0])
        # ax.set_ylim(bottom=xlim[0])

    if xlim[1] is None:
        ax.set_xlim(right=right)
    else:
        ax.set_xlim(right=xlim[1])
    ax.set_ylim(bottom=-0.95, top=0.8)
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    cbar = fig.colorbar(image)
    cbar.ax.tick_params(labelsize=15)
    fig.tight_layout()
    plt.suptitle(title)
    plt.show()

def scatter_hist(x, y, dot_color='black', label_r=False, identical_line=False,
    fit_line=False, exclude_zero=False, exclude_extremum=False, xlim=None, 
    ylim=None, xlabel=None, ylabel=None, title=None, fit_line_color='black',
    vline=None, hline=None):
    '''
    exclude_extremum: the exclude the value close to 1
    There are two overlapped histograms on the top and right side of the scatter plot
    '''
    not_nan = np.logical_and(np.logical_not(np.isnan(x)),
        np.logical_not(np.isnan(y)))

    good = not_nan
    if exclude_zero:
        not_zero = np.logical_and(x != 0, y != 0)
        good = np.logical_and(good, not_zero)
    if exclude_extremum:
        not_one = np.logical_and(abs(x) < 0.95, abs(y) < 0.95)
        good = np.logical_and(good, not_one)
    # print('number of samples: {};'.format(good.sum()))
    vmin = min(np.min(x[good]), np.min(y[good]))
    vmax = max(np.max(x[good]), np.max(y[good]))

    figsize = (6, 6)
    fig = plt.figure(figsize=(6, 6))

    fontsize = min(figsize) * 4
    plt.rcParams['font.size'] = fontsize

    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax_histx.spines[['top', 'right']].set_visible(False)
    ax_histy.spines[['top', 'right']].set_visible(False)

    # Plot scatter plot
    ax.scatter(x[good], y[good], s=2, c=dot_color, alpha=0.5)
    # ax.scatter(x[good], y[good], s=5, facecolors='none', edgecolors='r',
    # linewidths=1, alpha=0.5)
    
    # Plot polynomial fit
    x_fit = np.array((np.min(x[good]), np.max(x[good])))
    res = scipy.stats.linregress(x[good], y[good])
    if fit_line:
        ax.plot(x_fit, res.intercept + res.slope*x_fit, color=fit_line_color,
        label='Polyfit', linestyle='--')

    # coeffs = np.polyfit(x[good], y[good], deg=1)
    # poly = np.poly1d(coeffs)
    # x_fit = (np.min(x[good]), np.max(x[good]))
    # y_fit = poly(x_fit)
    # ax.plot(x_fit, y_fit, color='blue', label='Polyfit', linestyle='--')

    # Plot Pearson correlation coefficient
    corr_coef, _ = scipy.stats.pearsonr(x[good], y[good])
    _, pvalue = scipy.stats.ttest_rel(x[good], y[good])
    if label_r:
        ax.text(0.05, 0.95, 'r = {:.2f}'.format(corr_coef),
            transform=plt.gca().transAxes)
    # print('r = {:.2f}, p = {:.2e};'.format(corr_coef, pvalue))
    # print('x={:.3f}±{:.2e}, y={:.3f}±{:.2e};'.format(np.mean(x[good]),
    #     np.std(x[good]), np.mean(y[good]), np.std(y[good])))
    # print(scipy.stats.ttest_rel(x[good], y[good]))

    if identical_line:
        ax.plot(x_fit, x_fit, color='black', label='Identical', linestyle='--')
    # Set labels and legend
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax.set_xlabel('x', fontsize=fontsize)
    # ax.set_ylabel('y', fontsize=fontsize)
    # ax.legend(loc='upper right', fontsize=fontsize)
    
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        ax.set_xlim(left=vmin, right=vmax)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    else:
        ax.set_ylim(bottom=vmin, top=vmax)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if vline is not None:
        ax.axvline(vline, color='grey', linestyle='--')
    if hline is not None:
        ax.axhline(hline, color='grey', linestyle='--')

    # for the histgram ax_histx
    bins = 20
    _idx = np.logical_and(good, y > 0)
    ax_histx.hist(x[_idx], color='green', alpha=1, bins=bins,
        edgecolor='black', range=xlim)
    _idx = np.logical_and(good, y < 0)
    ax_histx.hist(x[_idx], color='orange', alpha=0.5, bins=bins,
        edgecolor='black', range=xlim)

    # for the histgram ax_histy
    _idx = np.logical_and(good, x > vline)
    ax_histy.hist(y[_idx], color='green', alpha=1, bins=bins,
        orientation='horizontal', edgecolor='black', range=ylim)
    _idx = np.logical_and(good, x < vline)
    ax_histy.hist(y[_idx], color='orange', alpha=0.5, bins=bins,
        orientation='horizontal', edgecolor='black', range=ylim)

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.suptitle(title, fontsize=fontsize)
    plt.show()

def scatter_polyfit(x, y, dot_color='black', label_r=False,identical_line=False,
    fit_line=False, exclude_zero=False, exclude_extremum=False, xlim=None, 
    ylim=None, xlabel=None, ylabel=None, title=None, fit_line_color='black'):
    '''
    exclude_extremum: the exclude the value close to 1
    '''
    not_nan = np.logical_and(np.logical_not(np.isnan(x)),
        np.logical_not(np.isnan(y)))

    good = not_nan
    if exclude_zero:
        not_zero = np.logical_and(x != 0, y != 0)
        good = np.logical_and(good, not_zero)
    if exclude_extremum:
        not_one = np.logical_and(abs(x) < 0.95, abs(y) < 0.95)
        good = np.logical_and(good, not_one)
    # print('number of samples: {};'.format(good.sum()))
    vmin = min(np.min(x[good]), np.min(y[good]))
    vmax = max(np.max(x[good]), np.max(y[good]))

    figsize = (5, 5)
    fig, ax = plt.subplots(figsize=figsize)
    fontsize = min(figsize) * 4
    plt.rcParams['font.size'] = fontsize
    # Plot scatter plot
    ax.scatter(x[good], y[good], s=2, c=dot_color, alpha=0.5)
    # ax.scatter(x[good], y[good], s=5, facecolors='none', edgecolors='r',
    # linewidths=1, alpha=0.5)
    
    # Plot polynomial fit
    x_fit = np.array((np.min(x[good]), np.max(x[good])))
    res = scipy.stats.linregress(x[good], y[good])
    if fit_line:
        ax.plot(x_fit, res.intercept + res.slope*x_fit, color=fit_line_color,
        label='Polyfit', linestyle='--')

    # coeffs = np.polyfit(x[good], y[good], deg=1)
    # poly = np.poly1d(coeffs)
    # x_fit = (np.min(x[good]), np.max(x[good]))
    # y_fit = poly(x_fit)
    # ax.plot(x_fit, y_fit, color='blue', label='Polyfit', linestyle='--')

    # Plot Pearson correlation coefficient
    corr_coef, _ = scipy.stats.pearsonr(x[good], y[good])
    _, pvalue = scipy.stats.ttest_rel(x[good], y[good])
    if label_r:
        ax.text(0.05, 0.95, 'r = {:.2f}'.format(corr_coef),
            transform=plt.gca().transAxes)
    # print('r = {:.2f}, p = {:.2e};'.format(corr_coef, pvalue))
    # print('x={:.3f}±{:.2e}, y={:.3f}±{:.2e};'.format(np.mean(x[good]),
    #     np.std(x[good]), np.mean(y[good]), np.std(y[good])))
    # print(scipy.stats.ttest_rel(x[good], y[good]))

    if identical_line:
        ax.plot(x_fit, x_fit, color='black', label='Identical', linestyle='--')
    # Set labels and legend
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax.set_xlabel('x', fontsize=fontsize)
    # ax.set_ylabel('y', fontsize=fontsize)
    # ax.legend(loc='upper right', fontsize=fontsize)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        ax.set_xlim(left=vmin, right=vmax)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    else:
        ax.set_ylim(bottom=vmin, top=vmax)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.suptitle(title, fontsize=fontsize)
    plt.show()

def scatter_bin(x, y, plot=True, title=None):
    '''
    x, y in the same shape
    x: the preferred orientation, in rang(0, 180)
    '''
    bin_width = 5
    x_range = (-90, 90)

    bins = int(180 / bin_width)
    x = x.flatten()
    y = y.flatten()
    bin_means, bin_edges, _ = scipy.stats.binned_statistic(x, y, statistic='mean', bins=bins, range=x_range)
    bin_counts, _, _ = scipy.stats.binned_statistic(x, y, statistic='count', bins=bins, range=x_range)
    # print(bin_counts)
    bin_stds, _, _ = scipy.stats.binned_statistic(x, y, statistic='std', bins=bins, range=x_range)
    bin_centers = bin_edges[:-1] + bin_width/2
    # print(bin_centers.shape)
    bin_sems = bin_stds / np.sqrt(bin_counts-1)
    if plot:
        figsize = (5, 5)
        fontsize = np.min(figsize) * 3
        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(x, y, s=1, c='black', alpha=0.5)
        ax.errorbar(bin_centers, bin_means, yerr=bin_sems, capsize=3)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='both', labelsize=fontsize)

        plt.title(title)
    return bin_centers, bin_means, bin_sems

def scatter_density(x, y, label_r=False, fit_line=False, identical_line=False, exclude_zero=False, exclude_extremum=False, title=None):
    '''
    exclude_extremum: the exclude the value close to 1
    '''
    not_nan = np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y)))
    good = not_nan
    if exclude_zero:
        not_zero = np.logical_and(x != 0, y != 0)
        good = np.logical_and(good, not_zero)
    if exclude_extremum:
        not_one = np.logical_and(x < 0.95, y < 0.95)
        good = np.logical_and(good, not_one)
    print('number of samples: {}'.format(good.sum()))

    # Calculate the point density
    x_good = x[good]
    print(x_good.max())
    y_good = y[good]
    xy_good = np.vstack([x[good], y[good]])
    z_good = scipy.stats.gaussian_kde(xy_good)(xy_good)
    
    # Sort the points by density, so that the densest points are plotted last
    idx = z_good.argsort()
    x_good, y_good, z_good = x_good[idx], y_good[idx], z_good[idx]

    figsize = (5, 5)
    fig, ax = plt.subplots(figsize=figsize)
    fontsize = min(figsize) * 3
    # Plot scatter plot
    ax.scatter(x_good, y_good, s=1, c=z_good, cmap='viridis')

    # Plot polynomial fit
    coeffs = np.polyfit(x[good], y[good], deg=1)
    poly = np.poly1d(coeffs)
    x_fit = (np.min(x[good]), np.max(x[good]))
    y_fit = poly(x_fit)
    if fit_line:
        ax.plot(x_fit, y_fit, color='blue', label='Polyfit', linestyle='--')

    # Plot Pearson correlation coefficient
    corr_coef, _ = scipy.stats.pearsonr(x[good], y[good])
    _, pvalue = scipy.stats.ttest_rel(x[good], y[good])
    print('r = {:.2f}'.format(corr_coef))
    print(scipy.stats.ttest_rel(x[good], y[good]))
    
    if label_r:
        ax.text(0.05, 0.95, 'r = {:.2f}, p = {:.2e}'.format(corr_coef, pvalue), transform=plt.gca().transAxes)

    if identical_line:
        ax.plot(x_fit, x_fit, color='red', label='Identical', linestyle='--')
    # Set labels and legend
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax.set_xlabel('x', fontsize=fontsize)
    # ax.set_ylabel('y', fontsize=fontsize)
    # ax.legend(loc='upper right', fontsize=fontsize)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.axvline(x=0, color='red', linestyle='--')
    ax.axhline(y=0, color='red', linestyle='--')
    plt.suptitle(title)
    plt.show()

def scatter_moving(x, y, thr=0.03, plot=True, title=None):
    '''
    Calculate the mean and standard error of the moving window scatter plot.

    x (array-like): The x-axis values.
    y (array-like): The y-axis values.
    thr (float, optional): The threshold value for filtering y-axis values. Defaults to 0.03.
    '''
    right = 90 # the  limit
    left, right = -90, 90 # the right, left limit
    good = y > thr
    x = x[good]
    y = y[good]
    window_center = np.arange(-90, 90)
    window = np.zeros((2, window_center.shape[0])) # the left and right limit of window
    window_width = 10
    window[0, :] = window_center - window_width/2
    window[1, :] = window_center + window_width/2
    window[0, :] = window[0, :] + (window[0, :] < left) * 180
    window[1, :] = window[1, :] - (window[1, :] > right) * 180
    mean = np.zeros(window_center.shape[0])
    sem = np.zeros(window_center.shape[0])
    for i in range(window.shape[1]):
        if window[0, i] < window[1, i]:
            mask = np.logical_and(x >= window[0, i], x < window[1, i])
        else:
            mask = np.logical_not(np.logical_and(x <= window[0, i], x > window[1, i]))
        mean[i] = np.mean(y[mask])
        sem[i] = scipy.stats.sem(y[mask])
    y1 = mean + sem
    y2 = mean - sem
    if plot:
        plt.fill_between(np.arange(window_center.shape[0]), y1, y2, color='black', alpha=0.3)
        plt.plot(mean, color='black')
        plt.plot
        plt.title(title)
        plt.show()
    return mean, sem


def scatter_hist_usual(x, y, dot_color='black', label_r=False, identical_line=False,
    fit_line=False, exclude_zero=False, exclude_extremum=False, xlim=None, 
    ylim=None, xlabel=None, ylabel=None, title=None, fit_line_color='black',
    vline=None, hline=None):
    '''
    exclude_extremum: the exclude the value close to 1
    The histograms are not onverlapped compare the 'scatter_hist()' function
    '''
    not_nan = np.logical_and(np.logical_not(np.isnan(x)),
        np.logical_not(np.isnan(y)))

    good = not_nan
    if exclude_zero:
        not_zero = np.logical_and(x != 0, y != 0)
        good = np.logical_and(good, not_zero)
    if exclude_extremum:
        not_one = np.logical_and(abs(x) < 0.95, abs(y) < 0.95)
        good = np.logical_and(good, not_one)
    # print('number of samples: {};'.format(good.sum()))
    vmin = min(np.min(x[good]), np.min(y[good]))
    vmax = max(np.max(x[good]), np.max(y[good]))

    figsize = (8, 8)
    fig = plt.figure(figsize=figsize)

    fontsize = 20
    plt.rcParams['font.size'] = fontsize

    ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
    ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
    ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    ax.spines[['top', 'right']].set_visible(False)
    ax_histx.spines[['top', 'right']].set_visible(False)
    ax_histy.spines[['top', 'right']].set_visible(False)

    # Plot scatter plot
    ax.scatter(x[good], y[good], s=20, alpha=1, marker='^', facecolors='none', 
        edgecolors=dot_color, linewidths=0.5)
    
    # Plot polynomial fit
    x_fit = np.array((np.min(x[good]), np.max(x[good])))
    res = scipy.stats.linregress(x[good], y[good])
    if fit_line:
        ax.plot(x_fit, res.intercept + res.slope*x_fit, color=fit_line_color,
        label='Polyfit', linestyle='--')

    # coeffs = np.polyfit(x[good], y[good], deg=1)
    # poly = np.poly1d(coeffs)
    # x_fit = (np.min(x[good]), np.max(x[good]))
    # y_fit = poly(x_fit)
    # ax.plot(x_fit, y_fit, color='blue', label='Polyfit', linestyle='--')

    # Plot Pearson correlation coefficient
    corr_coef, _ = scipy.stats.pearsonr(x[good], y[good])
    _, pvalue = scipy.stats.ttest_rel(x[good], y[good])
    if label_r:
        ax.text(0.05, 0.95, 'r = {:.2f}'.format(corr_coef),
            transform=plt.gca().transAxes)
    # print('r = {:.2f}, p = {:.2e};'.format(corr_coef, pvalue))
    x_mean = np.mean(x[good])
    x_std = np.std(x[good])
    y_mean = np.mean(y[good])
    y_std = np.std(y[good])
    # print('x={:.3f}±{:.2e}, y={:.3f}±{:.2e};'.format(x_mean, x_std, y_mean,
    #     y_std))
    # print(scipy.stats.ttest_rel(x[good], y[good]))
    
    ax.plot(x_mean, y_mean, marker='^', markersize=10, color=dot_color, alpha=1)
    p_left, p_right = x_mean - 0.5*x_std, x_mean + 0.5*x_std
    ax.plot([p_left, p_right], [y_mean, y_mean], color=dot_color, alpha=1,
        marker='|')
    p_bottom, p_top = y_mean - 0.5*y_std, y_mean + 0.5*y_std
    ax.plot([x_mean, x_mean], [p_bottom, p_top], color=dot_color, alpha=1,
        marker='_')

    if identical_line:
        ax.plot(x_fit, x_fit, color='black', label='Identical', linestyle='--')
    # Set labels and legend
    ax.tick_params(axis='both', labelsize=fontsize)
    # ax.set_xlabel('x', fontsize=fontsize)
    # ax.set_ylabel('y', fontsize=fontsize)
    # ax.legend(loc='upper right', fontsize=fontsize)
    
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        ax.set_xlim(left=vmin, right=vmax)
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    else:
        ax.set_ylim(bottom=vmin, top=vmax)
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    if vline is not None:
        ax.axvline(vline, color='grey', linestyle='--')
    if hline is not None:
        ax.axhline(hline, color='grey', linestyle='--')

    # for the histgram ax_histx
    bins = 20
    _idx = good
    ax_histx.hist(x[_idx], color='Orange', alpha=0.5, bins=bins,
        edgecolor='black', range=(-1,1))
    ax_histx.axvline(x=0, c='black', linestyle='--', linewidth=2)

    # for the histgram ax_histy
    _idx = good
    ax_histy.hist(y[_idx], color='green', alpha=0.5, bins=bins,
        orientation='horizontal', edgecolor='black', range=(-1,1))
    ax_histy.axhline(y=0, c='black', linestyle='--', linewidth=2)

    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.suptitle(title, fontsize=fontsize)
    plt.show()

def ax_plot_distanceRFcenter_si(ax, distance_rf_center, si, cell_type, x_max=None, idx_snr=None, bk_pref=None, bar_color='lightgrey', sub_title=None):
    # plot the distance to RF center VS si
    idx_not_nan = np.logical_and(np.logical_not(np.isnan(si)), np.logical_not(np.isnan(distance_rf_center)))
    idx_good = np.logical_and(idx_not_nan, np.abs(si)!=1)
    idx_good_type = np.logical_and(idx_good, cell_type)
    if idx_snr is not None:
        # print('select the rois base on snr')
        idx_good_type = np.logical_and(idx_good_type, idx_snr)
    if bk_pref is not None:
        # print('select the rois base on bk_pref')
        idx_good_type = np.logical_and(idx_good_type, bk_pref)
    
    n_sample = idx_good_type[0, 0, :].sum()
    print('number of samples: {}'.format(n_sample))
    x = distance_rf_center[idx_good_type].flatten()
    if x_max == None:
        x_max = np.max(x)

    y = si[idx_good_type].flatten()

    bin_width = 5
    bin_n = np.ceil(x_max / bin_width).astype(int)
    mean = np.zeros(bin_n)
    std = np.zeros(bin_n)
    sem = np.zeros(bin_n)

    for i in range(bin_n):
        idx_sel = np.logical_and(i*bin_width <= x, x < (i+1)*bin_width)
        mean[i] = np.mean(y, where=idx_sel)
        std[i] = np.std(y, where=idx_sel)
        sem[i] = scipy.stats.sem(y[idx_sel])
    
    # fig, ax = plt.subplots()
    ax.bar(x=(np.arange(bin_n) + 1/2)*bin_width, height=mean, yerr=sem, alpha=1.0, color=bar_color, capsize=5, edgecolor='black', width=bin_width)
    # ax.set_xlabel('Distance to RF center')
    # ax.set_ylabel('SI')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(sub_title)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    return mean, std, sem, n_sample

def ax_plot_distanceRFcenter_v2(ax, distance_rf_center, data, x_max=50, idx_sel=None, bar_color='gray', sub_title=None):
    '''
    plot the distance to RF center VS data.
    distance_rf_center: shape in (4, 6, n_rois)
    data: shape in (4, 6, n_rois), same as distance_rf_center
    '''
    # plot the distance to RF center VS data
    idx_good = np.logical_not(np.isnan(distance_rf_center[0,0,:]))
    # idx_good = np.logical_and(idx_not_nan, np.abs(data)!=1)
    if idx_sel is not None:
        idx_good = np.logical_and(idx_good, idx_sel)
    
    idx_good_indx = np.where(idx_good)[0]
    # print('number of rois: {}'.format(idx_good.sum()))
    
    distance_rf_center_sel = distance_rf_center[:,:,idx_good]
    data_sel = data[:,:,idx_good]
    # print('data_sel shape: {}'.format(data_sel.shape))
    
    # the response of the position that is nearest to the rf center
    salient_resp = np.empty(data.shape[2])
    salient_resp[:] = np.nan
    # print('salient_resp shape: {}'.format(salient_resp.shape))
    for i in range(data_sel.shape[2]):
        _min_idx = min_idx_2d(distance_rf_center_sel[:,:,i])
        salient_resp[idx_good_indx[i]] = data_sel[_min_idx[0], _min_idx[1], i]
    
    bin_width = 5
    bin_n = np.ceil(x_max / bin_width).astype(int)
    mean = np.zeros(bin_n)
    std = np.zeros(bin_n)
    sem = np.zeros(bin_n)

    for i in range(bin_n):
        idx_sel = np.logical_and(distance_rf_center_sel>=i*bin_width, distance_rf_center_sel<(i+1)*bin_width)
        mean[i] = np.mean(data_sel[idx_sel])
        std[i] = np.std(data_sel[idx_sel])
        sem[i] = scipy.stats.sem(data_sel[idx_sel])
    
    # ax.bar(x=(np.arange(bin_n) + 1/2)*bin_width, height=mean, yerr=sem, width=bin_width, color=bar_color, edgecolor='black', linewidth=1.0, capsize=5)

    facecolor = mcolors.to_rgb(bar_color)
    facecolor_alpha = (facecolor[0], facecolor[1], facecolor[2], 0.5)
    ax.bar(x=(np.arange(bin_n) + 1/2)*bin_width, height=mean, yerr=sem, width=bin_width, facecolor=facecolor_alpha, edgecolor='black', linewidth=1.0, capsize=5)

    # ax.set_xlabel('Distance to RF center')
    # ax.set_ylabel('SI')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_title(sub_title)
    return salient_resp

def plot_distanceRFcenter_si(distance_rf_center, si, cell_type, x_max=None, bar_color ='lightgray', idx_snr=None, bk_pref=None, title=None):
    # plot the distance to RF center VS si
    idx_not_nan = np.logical_and(np.logical_not(np.isnan(si)), np.logical_not(np.isnan(distance_rf_center)))
    idx_good = np.logical_and(idx_not_nan, np.abs(si)!=1)
    idx_good_type = np.logical_and(idx_good, cell_type)
    if idx_snr is not None:
        # print('select the rois base on snr')
        idx_good_type = np.logical_and(idx_good_type, idx_snr)
    if bk_pref is not None:
        # print('select the rois base on bk_pref')
        idx_good_type = np.logical_and(idx_good_type, bk_pref)
    
    x = distance_rf_center[idx_good_type].flatten()
    if x_max == None:
        x_max = np.max(x)

    y = si[idx_good_type].flatten()

    bin_width = 5
    bin_n = np.ceil(x_max / bin_width).astype(int)
    mean = np.zeros(bin_n)
    std = np.zeros(bin_n)
    sem = np.zeros(bin_n)

    for i in range(bin_n):
        idx_sel = np.logical_and(i*bin_width <= x, x < (i+1)*bin_width)
        mean[i] = np.mean(y, where=idx_sel)
        std[i] = np.std(y, where=idx_sel)
        sem[i] = scipy.stats.sem(y[idx_sel])
    
    fig, ax = plt.subplots()
    ax.bar(x=(np.arange(bin_n) + 1/2)*bin_width, height=mean, yerr=sem, width=bin_width, color=bar_color, edgecolor='black', capsize=5)
    # ax.set_xlabel('Distance to RF center')
    # ax.set_ylabel('SI')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.suptitle(title)
    
def plot_distanceRFcenter_v2(distance_rf_center, data, cell_type, x_max=50, bk_pref=None, title=None):
    '''
    plot the distance to RF center VS data.
    distance_rf_center: shape in (4, 6, n_rois)
    data: shape in (4, 6, n_rois), same as distance_rf_center
    '''
    # plot the distance to RF center VS data
    idx_not_nan = np.logical_not(np.isnan(distance_rf_center[0,0,:]))
    # idx_good = np.logical_and(idx_not_nan, np.abs(data)!=1)
    idx_good_type = np.logical_and(idx_not_nan, cell_type)
    if bk_pref is not None:
        idx_good_type = np.logical_and(idx_good_type, bk_pref)
    
    print(np.sum(idx_good_type))
    print('number of rois: {}'.format(idx_good_type.sum()))
    
    distance_rf_center_sel = distance_rf_center[:,:,idx_good_type]
    data_sel = data[:,:,idx_good_type]
    # print('data_sel shape: {}'.format(data_sel.shape))
    
    # the response of the position that is nearest to the rf center
    salient_resp = np.zeros(data_sel.shape[2])
    # print('salient_resp shape: {}'.format(salient_resp.shape))
    for i in range(data_sel.shape[2]):
        _min_idx = min_idx_2d(distance_rf_center_sel[:,:,i])
        salient_resp[i] = data_sel[_min_idx[0], _min_idx[1], i]
    
    bin_width = 5
    bin_n = np.ceil(x_max / bin_width).astype(int)
    mean = np.zeros(bin_n)
    std = np.zeros(bin_n)
    sem = np.zeros(bin_n)

    for i in range(bin_n):
        idx_sel = np.logical_and(distance_rf_center_sel>=i*bin_width, distance_rf_center_sel<(i+1)*bin_width)
        mean[i] = np.mean(data_sel[idx_sel])
        std[i] = np.std(data_sel[idx_sel])
        sem[i] = scipy.stats.sem(data_sel[idx_sel])
    
    fig, ax = plt.subplots()
    ax.bar(x=(np.arange(bin_n) + 1/2)*bin_width, height=mean, yerr=sem, width=bin_width, color='lightgrey', edgecolor='black', capsize=5)
    # ax.set_xlabel('Distance to RF center')
    # ax.set_ylabel('SI')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.suptitle(title)
    plt.show()
    return salient_resp

def pie_customize(data, si_thr=0, colors=['lightcoral', 'lightgray']):
    not_nan = np.logical_not(np.isnan(data))
    higher = np.logical_and(data > si_thr, not_nan)
    # lower = np.logical_and(data < -si_thr, not_nan)
    size_higher = higher.sum()
    # size_lower = lower.sum()
    size_other = not_nan.sum() - size_higher
    sizes = [size_higher, size_other] 

    percentage_higher = sizes[0] / np.array(sizes).sum()
    percentage_other = sizes[1] / np.array(sizes).sum()
    # percentage_lower = sizes[2] / np.array(sizes).sum()
    # print('{:.0%}, {:.0%}'.format(percentage_higher, percentage_other))

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.pie(sizes, colors=colors, autopct='%1.0f%%', textprops={'fontsize':20},
        wedgeprops={"alpha": 0.5})
    # ax.pie(sizes, colors=colors)
    ax.texts[3].remove()
    ax.texts[1].remove()
    fig.tight_layout()
    plt.show()

def pie_customize_stacked(data1, data2, si_thr=0):
    not_nan = np.logical_not(np.isnan(data1))
    colors=['lightcoral', 'lightgray']
    higher1 = np.logical_and(data1 > si_thr, not_nan)
    higher2 = np.logical_and(data2 > si_thr, not_nan)
    size_higher = higher1.sum() + higher2.sum()
    # size_lower = lower.sum()
    size_other = not_nan.sum()*2 - size_higher
    sizes = [size_higher, size_other] 

    percentage_higher = sizes[0] / np.array(sizes).sum()
    percentage_other = sizes[1] / np.array(sizes).sum()
    # percentage_lower = sizes[2] / np.array(sizes).sum()
    print('{:.0%}, {:.0%}'.format(percentage_higher, percentage_other))

    fig, ax = plt.subplots(figsize=(2, 2))
    # ax.pie(sizes, labels=labels, autopct='%1.2f%%', textprops={'fontsize': 15})
    ax.pie(sizes, colors=colors, autopct='%1.0f%%', textprops={'fontsize':25})
    fig.tight_layout()
    plt.show()

def plot_rf_vs_dist_batch(distance_rf_center, sg1_resp, bk_resp, cell_type, x_max, bk_pref, idx_snr, title_prefix):
    
    si = cal_si(sg1_resp, bk_resp)

    plot_distanceRFcenter_si(distance_rf_center, sg1_resp, cell_type=cell_type,  x_max=x_max, idx_snr=idx_snr, bk_pref=bk_pref, title=title_prefix + '_Resp')
    
    plot_distanceRFcenter_si(distance_rf_center, np.broadcast_to(bk_resp, sg1_resp.shape), cell_type=cell_type,  x_max=x_max, idx_snr=idx_snr, bk_pref=bk_pref, title=title_prefix + '_BG')
    
    plot_distanceRFcenter_si(distance_rf_center, si, cell_type=cell_type,  x_max=x_max, idx_snr=idx_snr, bk_pref=bk_pref, title=title_prefix + '_SI')

def cal_distance_rf_center(rf_pos_deg_fit, visual_pos):
    n_rois = rf_pos_deg_fit.shape[-1]
    distance_rf_center = np.ones((4, 6, n_rois))*np.nan
    for roi in range(n_rois):
        for col in range(visual_pos.shape[1]):
            for row in range(visual_pos.shape[0]):
                # there are Nan in rf['rf_pos_deg_fit']
                distance_rf_center_xy = visual_pos[row, col, :] - rf_pos_deg_fit[:, roi]
                distance_rf_center[row, col, roi] = np.linalg.norm(distance_rf_center_xy)
    return distance_rf_center

def roi_type(all_data, parent_folder):
    strain = {'Vglut2': ['C49', 'C50', 'C51', 'C51_saline', 'C51_DOM'],
        'Vgat': ['C52', 'C53', 'C54', 'C55', '#C449']}
    is_exc_ls = []
    is_inh_ls = []
    for mice in all_data.keys():
        if mice in strain['Vglut2']:
            gen_type = 'Vglut2'
        if mice in strain['Vgat']:
            gen_type = 'Vgat'
        for date in all_data[mice].keys():
            folder_2p_save_path = parent_folder + "two-photon/" + mice + '/' + date
            for plane in all_data[mice][date]:
                results_path = folder_2p_save_path + '/plane' + str(plane) + '/analysis/results.hdf5'
                rois_channel_dic = h5py_read(results_path, group_read='results_rf', dataset_read='rois_channel')
                rois_channel = rois_channel_dic['results_rf']['rois_channel']
                if gen_type == 'Vglut2':
                    is_inh = (rois_channel == 2)
                    is_exc = np.logical_not(is_inh)
                if gen_type == 'Vgat':
                    is_exc = (rois_channel == 2)
                    is_inh = np.logical_not(is_exc)
                is_exc_ls.append(is_exc)
                is_inh_ls.append(is_inh)

    return is_exc_ls, is_inh_ls

def read_data(all_data, group_name, dataset_name, parent_folder):
    data_ls = []
    for mice in all_data.keys():
        for date in all_data[mice].keys():
            folder_2p_save_path = parent_folder + "two-photon/" + mice + '/' + date
            for plane in all_data[mice][date]:
                results_path = folder_2p_save_path + '/plane' + str(plane) + '/analysis/results.hdf5'
                with h5py.File(results_path, "a") as f:
                    if '{}/{}'.format(group_name, dataset_name) in f:
                        print('Reading {} {} in {}'.format(group_name, dataset_name, results_path))
                        _data = h5py_read(results_path, group_read=group_name, dataset_read=dataset_name)
                        data_ls.append(_data[group_name][dataset_name])
                    else:
                        print('No {} {} in {}'.format(group_name, dataset_name, results_path))
                        print('A \'None\' was added to data_ls.')
                        data_ls.append(None)
    return data_ls

# def read_data_suite2p(all_data, group_name, dataset_name, parent_folder):
#     '''
#     Read data of hdf5 file in suite2p folder. 
#     '''

#     data_ls = []
#     for mice in all_data.keys():
#         for date in all_data[mice].keys():
#             exp_folder = parent_folder + "two-photon/" + mice + '/' + date
#             for plane in all_data[mice][date]:
#                 plane_folder = exp_folder + '/suite2p/plane' + str(plane)
#                 analysis_folder = plane_folder + '/analysis'
#                 results_path = os.path.join(analysis_folder, 'results.hdf5')
#                 with h5py.File(results_path, "a") as f:
#                     if '{}/{}'.format(group_name, dataset_name) in f:
#                         print('Reading {} {} in {}'.format(group_name, dataset_name, results_path))
#                         _data = h5py_read(results_path, group_read=group_name, dataset_read=dataset_name)
#                         data_ls.append(_data[group_name][dataset_name])
#                     else:
#                         print('No {} {} in {}'.format(group_name, dataset_name, results_path))
#                         print('A \'None\' was added to data_ls.')
#                         data_ls.append(None)
#     return data_ls

def read_data_suite2p(all_data, group_name, dataset_name, parent_folder,
    treatment=None):
    '''

    Read data of hdf5 file in suite2p folder. 
    '''

    data_ls = []
    for mice in all_data.keys():
        for date in all_data[mice].keys():
            exp_folder = parent_folder + "two-photon/" + mice + '/' + date
            for plane in all_data[mice][date]:
                plane_folder = exp_folder + '/suite2p/plane' + str(plane)
                analysis_folder = plane_folder + '/analysis'
                if treatment is None:
                    results_path = os.path.join(analysis_folder, 'results.hdf5')
                else:
                    results_path = os.path.join(analysis_folder,
                        'results_{}.hdf5'.format(treatment))
                print('Reading {}'.format(results_path))
                _data = h5py_read(results_path, group_read=group_name, dataset_read=dataset_name)
                # if _data is not None:
                #     print('Readed {} {} in {}'.format(group_name, dataset_name, results_path))
                # else:
                #     print('No {} {} in {}'.format(group_name, dataset_name, results_path))
                #     print('A \'None\' was added to data_ls.')

                data_ls.append(_data[group_name][dataset_name])

    return data_ls


def read_behavior_data(all_data, data_name):
    '''
    read the data of behavior
    data_name: the name of the data to read, available data_name: 'speed', 'Pupil_area', 'Pupil_X', 'Pupil_Y'
    '''
    parent_folder = 'Z:/saliency_map/'
    mean_ls = []
    data_ls =[]
    for mice in all_data.keys():
        for date in all_data[mice].keys():
            folder_behavioral_path = parent_folder + "behavioral_data/" + mice + '/' + date
            stimulus_ls, _ = get_stimulus_ls(exp_date=date)
            # print(folder_behavioral_path)
            stimuli_list = list(range(1, len(stimulus_ls) + 1))
            mean = np.zeros(len(stimuli_list))
            for stim in stimuli_list:
                # the file path of behavior data
                fpath_hdf5 = folder_behavioral_path + "/results/{}_{}_stimuli_{}.hdf5".format(mice, date, stim)
                # print(fpath_hdf5)
                _temp = h5py_read(fpath_hdf5, group_read='behavior',dataset_read=data_name)
                _temp = _temp['behavior'][data_name] # dict to array
                # print(_temp.shape)
                _temp = np.abs(_temp)
                data_ls.append(_temp)
                mean[stim-1] = np.mean(_temp) # the mean of the absolute value
            mean_ls.append(mean)
    return mean_ls, data_ls

def get_n_exp(all_data):
    # get the number of experiments
    n_exp = 0
    for mice in all_data.keys():
        for date in all_data[mice].keys():
            # print(len(all_data[mice][date]))
            n_exp = n_exp + len(all_data[mice][date])
    return n_exp

def get_exp_info(all_data, exp_idx):
    '''
    get the mice_id, date, plane of the exp_idx
    '''
    n_exp = get_n_exp(all_data)
    if exp_idx >= n_exp:
        print('exp index out of range')
        return None
    exp = 0
    for mice_id in all_data.keys():
        for date in all_data[mice_id].keys():
            for plane in all_data[mice_id][date]:
                if exp == exp_idx:
                    return mice_id, date, plane
                exp = exp + 1

def get_bk_resp(all_data, parent_folder):
    '''
    get the background response of SG1.
    if background response of sg0 does not exit, use the sg2
    # the function has been tested is correct
    '''
    bk_resp_0 = []
    bk_resp_90 = []
    for mice in all_data.keys():
        for date in all_data[mice].keys():
            folder_2p_save_path = parent_folder + "two-photon/" + mice + '/' + date
            for plane in all_data[mice][date]:
                results_path = folder_2p_save_path + '/plane' + str(plane) + '/analysis/results.hdf5'
                # get the number of rois
                n_rois_dic = h5py_read(results_path, group_read='results_rf', dataset_read='n_rois')
                n_rois = n_rois_dic['results_rf']['n_rois']

                print('Reading background response in {}'.format(results_path))
                try:
                    _data = h5py_read(results_path, group_read='results_sg_0', dataset_read='mean_amp')
                    _data = _data['results_sg_0']['mean_amp']
                    _data_0 = _data[0, -1, :n_rois]
                    _data_90 = _data[3, -1, :n_rois]
                except KeyError:
                    _data = h5py_read(results_path, group_read='results_sg_2', dataset_read='mean_amp')
                    _data = _data['results_sg_2']['mean_amp']
                    _data_0 = _data[0, -1, :n_rois]
                    _data_90 = _data[6, -1, :n_rois]

                bk_resp_0.append(_data_0)
                bk_resp_90.append(_data_90)
    return bk_resp_0, bk_resp_90

def get_visual_pos(stim_center, n_rows=4, n_cols=6, unit_size=10):

    # the left top corner of the saliency stimuli
    visual_origin = np.array([stim_center[0] - n_cols / 2*unit_size, stim_center[1] + n_rows / 2*unit_size])

    # calculate the position of salience position
    visual_pos = np.zeros((n_rows, n_cols, 2))
    for i in range(n_rows):
        for j in range(n_cols):
            _row_pos = visual_origin[1]-i*unit_size - unit_size/2
            _col_pos = visual_origin[0]+j*unit_size + unit_size/2
            visual_pos[i,j,0] = _col_pos
            visual_pos[i,j,1] = _row_pos

    return visual_pos

from PIL import Image
def fig2img(fig):
    # Draw the content
    fig.canvas.draw()
    
    # Get the RGB values
    rgb = fig.canvas.tostring_rgb()
    
    # Get the width and height of the figure
    width, height = fig.canvas.get_width_height()
    
    # Convert the RGB values to a PIL Image
    img = Image.frombytes('RGB', (width, height), rgb)
    
    # Convert the PIL Image to a Numpy array
    img_array = np.array(img)
    return img_array

# import cv2    
# def img2video(images,video_path,fps=30):    
#     height, width, layers = images[0].shape    
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
#     video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
#     for image in images:
#         video.write(image)
        
#     cv2.destroyAllWindows()
#     video.release()

def violin_plot(data, facecolor='orange', labels=None, alpha=0.5):
    fig, ax = plt.subplots(figsize=(4, 5))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor('black')
        pc.set_alpha(alpha)

    for i in range(len(data)):
        quartile1, medians, quartile3 = np.percentile(data[i], [25, 50, 75], axis=0)
        sorted_array = np.sort(data[i])
        whisker = adjacent_values(sorted_array, quartile1, quartile3)
        mean = np.mean(data[i])

        ax.scatter(i+1, medians, marker='o', color='white', s=30, zorder=3)
        ax.scatter(i+1, mean, marker='_', color='cyan', s=200, zorder=3)
        ax.vlines(i+1, quartile1, quartile3, color='k', linestyle='-', lw=7)
        ax.vlines(i+1, whisker[0], whisker[1], color='k', linestyle='-', lw=1)

    bars_loc = np.arange(1, len(data)+1)
    ax.set_xticks(bars_loc, labels)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.spines[['top', 'right']].set_visible(False)
    plt.show()
