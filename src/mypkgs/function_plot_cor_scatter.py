import numpy as np
import xarray as xr
import pandas as pd
import os, time, dask
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy import stats

def plot_corr_scatter(ax, x_data, y_data, title, sequence, xlabel, ylabel):
    # 线性回归分析
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
    # 扩展自变量范围
    extended_delta_u = np.linspace(min(x_data) - 1, max(x_data) + 1, 100)
    # 计算扩展范围内的回归值
    extended_line = slope * extended_delta_u + intercept

    years = np.arange(1979, 2025)
    marker_list = [f'${year}$' for year in years]
    ax.plot(extended_delta_u, extended_line, color='red', linewidth=1.5)
    for x, y, year in zip(x_data, y_data, years):
        if year < 1990:
            color = '#5edc1f'  #
        elif year < 2000:
            color = '#274afd'
        elif year < 2010:
            color = '#fc86aa'  # pinky
        else:
            color = '#9a0eea'  #
        ax.scatter(x, y, s=200, color=color, marker=f'${year}$',
                   zorder=20, linewidths=0.1, alpha=0.6)
    # 添加图例、标签和标题
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    if sequence == "c" or "e":
        xy_loc = (0.15, 0.25)
    else:
        xy_loc = (0.15, 0.25)

    if p_value < 0.01:
        ax.annotate(f'R = {r_value:.2f}\np < 0.01', xy=xy_loc, xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top', fontsize=9)
    elif p_value < 0.05:
        ax.annotate(f'R = {r_value:.2f}\np < 0.05', xy=xy_loc, xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top', fontsize=9)
    else:
        ax.annotate(f'R = {r_value:.2f}\np = {p_value:.2f}', xy=xy_loc, xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top', fontsize=9)

    # ax.vlines(0, -2, 5, colors='grey', linestyles='--', linewidths=.5)
    # ax.hlines(0, -8, 12, colors='grey', linestyles='--', linewidths=.5)
    # 增大坐标轴标签和刻度的字体大小
    ax.tick_params(axis='both', which='major', direction='in')  # labelsize=7,
    # ax.set_title('Based on T', loc='left', fontsize=14)
    ax.set_title(title, loc='right', fontsize=10, pad=4)  # , fontsize=7
    ax.text(-0.13, 1.17, sequence, transform=ax.transAxes, fontsize=18, fontweight='bold', va='top', ha='left')
    return ax