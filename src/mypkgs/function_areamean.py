# -*- coding: utf-8 -*-
"""
Created on Thu Sep 29 21:36:22 2022
"""

import numpy as np
import xarray as xr
import copy


def am(ds):
    '''
    do the latitude of the weighted and calculate the Area mean value
    Parameters
    ----------
    ds : Xarray.Dataarray(t,lat,lon)
        input: 3D data

    Returns
    -------
    ds_lwm : local mean of the
        output: Area mean value

    '''
    lat_label = ds.dims[1]
    lon_label = ds.dims[2]
    nrows = len(ds[lat_label])  # 纬度数组长度
    ncols = len(ds[lon_label])  # 经度数组长度
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))  # 将度转化为pi
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)  # 设置权重矩阵
    #    ds_lw = copy.deepcopy(ds)                                  #复制一个大小相同的dataarray
    ds_lw = ds * weight_matrix  # 对相应时间和模式做纬度加权
    ds_lwm = ds_lw.sum(dim=[lat_label, lon_label]) / np.sum(weight_matrix)  # 计算全球平均
    return ds_lwm


def am1D(ds):
    '''
    do the latitude of the weighted and calculate the Area mean value
    Parameters
    ----------
    ds : Xarray.Dataarray(lat,lon)
        input: 1D data

    Returns
    -------
    ds_lwm : local mean of the
        output: Area mean value

    '''
    lat_label = ds.dims[0]
    lon_label = ds.dims[1]
    nrows = len(ds[lat_label])  # 纬度数组长度
    ncols = len(ds[lon_label])  # 经度数组长度
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))  # 将度转化为pi
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)  # 设置权重矩阵
    # ds_lw = copy.deepcopy(ds)                                  #复制一个大小相同的dataarray
    ds_lw = ds * weight_matrix  # 对相应时间和模式做纬度加权
    ds_lwm = ds_lw.sum(dim=[lat_label, lon_label]) / np.sum(weight_matrix)  # 计算全球平均
    return ds_lwm


def am4D(ds):
    model_label, time_label, lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ds * weight_matrix
    ds_mean = ds_w.sum(dim=[lon_label, lat_label]) / np.sum(weight_matrix)
    return ds_mean


def am5D(ds):
    months_label, model_label, level_label, lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ds * weight_matrix
    ds_mean = ds_w.sum(dim=[lon_label, lat_label]) / np.sum(weight_matrix)
    return ds_mean


def nan_lev_am(ds):
    lev_label, lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    nlev = len(ds[lev_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_gl = np.zeros(nlev)
    ds_w = ds * weight_matrix
    for i in range(nlev):
        weight_sum = copy.deepcopy(weight_matrix)
        if ds_w[i].isnull().any():
            weight_sum[ds_w[i].isnull()] = np.nan
        ds_gl[i] = ds_w[i].sum(dim=[lon_label, lat_label]) / np.nansum(weight_sum)
    return ds_gl


def nan_am(ds):
    time_label, lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    ntime = len(ds[time_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_gl = np.zeros(ntime)
    ds_w = ds * weight_matrix
    for i in range(ntime):
        weight_sum = copy.deepcopy(weight_matrix)
        if ds_w[i].isnull().any():
            weight_sum[ds_w[i].isnull().data] = np.nan
            ds_gl[i] = ds_w[i].sum(dim=[lon_label, lat_label]) / np.nansum(weight_sum)
    ds_mean = xr.DataArray(ds_gl, dims=('time'), coords={'time': ds[time_label]})
    return ds_mean

def nan_models_times_levels_am_5D(ds):
    model_label, time_label,level_label, lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    nmodels = len(ds[model_label])
    ntimes = len(ds[time_label])
    nlevels = len(ds[level_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_gl = np.zeros((nmodels,ntimes,nlevels))
    ds_w = ds * weight_matrix
    for j in range(nmodels):
        for i in range(nlevels):
            weight_sum = copy.deepcopy(weight_matrix)
            if ds_w[j,:,i].isnull().any():
                weight_sum[ds_w[j,:,i].isnull().data] = np.nan
                ds_gl[j,:,i] = ds_w[j,:,i].sum(dim=[lon_label, lat_label]) / np.nansum(weight_sum)
    ds_mean = xr.DataArray(ds_gl, dims=('model','time','level'),
                           coords={'model': ds[model_label],'time': ds[time_label],'level': ds[level_label]})
    return ds_mean

def nan_models_times_levels_am_4D(ds):
    time_label,level_label, lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    ntimes = len(ds[time_label])
    nlevels = len(ds[level_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_gl = np.zeros((ntimes,nlevels))
    ds_w = ds * weight_matrix
    for i in range(nlevels):
        weight_sum = copy.deepcopy(weight_matrix)
        if ds_w[:,i].isnull().any():
            weight_sum[ds_w[:,i].isnull().data] = np.nan
            ds_gl[:,i] = ds_w[:,i].sum(dim=[lon_label, lat_label]) / np.nansum(weight_sum)
    ds_mean = xr.DataArray(ds_gl, dims=('time','plev'),
                           coords={'time': ds[time_label],'plev': ds[level_label]})
    return ds_mean

def mask_am(ds):
    time_label, lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    ntime = len(ds[time_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ds * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    weight_sum[ds_w[0].isnull().data] = np.nan
    ds_mean = ds_w.sum(dim=[lon_label, lat_label]) / np.nansum(weight_sum)
    return ds_mean


def mask_am1D(ds):
    lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ds * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    weight_sum[ds_w.isnull().data] = np.nan
    ds_mean = ds_w.sum(dim=[lon_label, lat_label]) / np.nansum(weight_sum)
    return ds_mean


def mask_am4D(ds):
    model_label, time_label, lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ds * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    weight_sum[ds_w[0, 0].isnull().data] = np.nan
    ds_mean = ds_w.sum(dim=[lon_label, lat_label]) / np.nansum(weight_sum)
    return ds_mean

def mask_am5D(ds):
    months_label, model_label, time_label, lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ds * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    weight_sum[ds_w[0, 0, 0].isnull().data] = np.nan
    ds_mean = ds_w.sum(dim=[lon_label, lat_label]) / np.nansum(weight_sum)
    return ds_mean



def mask_am6D(ds):
    months_label, model_label, time_label, level_label,lat_label, lon_label = ds.dims
    nrows = len(ds[lat_label])
    ncols = len(ds[lon_label])
    latsr = np.deg2rad(ds[lat_label].values).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ds * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    weight_sum[ds_w[0, 0, 0, 0].isnull().data] = np.nan
    ds_mean = ds_w.sum(dim=[lon_label, lat_label]) / np.nansum(weight_sum)
    return ds_mean
####################upward shift of jet stream###############
def areamean_func_5D(ndarray,slice,lat_ds):
    months_label, time_label, level_label, lat_label,lon_label = ndarray.shape
    nrows = lat_label
    ncols = lon_label
    latsr = np.deg2rad(lat_ds['lat'].values[slice]).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ndarray * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    #weight_sum[ds_w[0, 0, 0].isnull().data] = np.nan
    obs_mean_areamean = ds_w.sum(axis=3).sum(axis=3)/ np.nansum(weight_sum)
    return obs_mean_areamean
def areamean_func_4D(ndarray,slice,lat_ds):
    months_label, time_label, lat_label,lon_label = ndarray.shape
    nrows = lat_label
    ncols = lon_label
    latsr = np.deg2rad(lat_ds['lat'].values[slice]).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ndarray * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    #weight_sum[ds_w[0, 0].isnull().data] = np.nan
    obs_mean_areamean = ds_w.sum(axis=2).sum(axis=2) / np.nansum(weight_sum)
    return obs_mean_areamean

def areamean_func_3D_mask_nan(ndarray,slice,lat_ds):
    time_label, lat_label,lon_label = ndarray.shape
    nrows = lat_label
    ncols = lon_label
    latsr = np.deg2rad(lat_ds['lat'].values[slice]).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ndarray * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    #weight_sum[ds_w[0, 0].isnull().data] = np.nan
    #weight_sum[np.isnan(ds_w[0, 0])] = np.nan
    #bs_mean_areamean = ds_w.sum(axis=1).sum(axis=1) / np.nansum(weight_sum)
    obs_mean_areamean = np.nansum(np.nansum(ds_w, axis=1), axis=1) / np.nansum(weight_sum)
    return obs_mean_areamean
def areamean_func_2D_mask_nan(ndarray,slice,lat_ds):
    lat_label,lon_label = ndarray.shape
    nrows = lat_label
    ncols = lon_label
    latsr = np.deg2rad(lat_ds['lat'].values[slice]).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), ncols, axis=1)
    ds_w = ndarray * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    #weight_sum[ds_w.isnull().data] = np.nan#####
    weight_sum[np.isnan(ds_w)] = np.nan
    #ds_w_nanto0 = np.nan_to_num(ds_w)
    #obs_mean_areamean = ds_w_nanto0.sum(axis=0).sum(axis=0) / np.nansum(weight_sum)
    obs_mean_areamean = np.nansum(np.nansum(ds_w,axis=0),axis=0) / np.nansum(weight_sum)
    return obs_mean_areamean

##########南半球40:56，北半球89:105############
def areamean_func_slice(ndarray,slice,lat_ds):
    months_label, time_label, level_label, lat_label,lon_label = ndarray.shape
    nrows = lat_label
    latsr = np.deg2rad(lat_ds['lat'].values[slice]).reshape((nrows, 1))
    #weight_matrix = np.repeat(np.cos(latsr), 1, axis=1)
    ds_w = ndarray * np.cos(latsr)
    weight_sum = copy.deepcopy(np.cos(latsr))
    #weight_sum[ds_w[0, 0, 0].isnull().data] = np.nan
    obs_mean_areamean = ds_w.sum(3) / np.nansum(weight_sum)
    return obs_mean_areamean
def areamean_func_4D_slice(ndarray,slice,lat_ds):
    months_label, time_label, lat_label,lon_label = ndarray.shape
    nrows = lat_label
    latsr = np.deg2rad(lat_ds['lat'].values[slice]).reshape((nrows, 1))
    #weight_matrix = np.repeat(np.cos(latsr), 1, axis=1)
    ds_w = ndarray * np.cos(latsr)
    weight_sum = copy.deepcopy(np.cos(latsr))
    #weight_sum[ds_w[0, 0, 0].isnull().data] = np.nan
    obs_mean_areamean = ds_w.sum(2) / np.nansum(weight_sum)
    return obs_mean_areamean