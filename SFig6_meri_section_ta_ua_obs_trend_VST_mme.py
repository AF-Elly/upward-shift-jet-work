import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import stats
from scipy import optimize
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from sklearn.feature_selection import f_regression
import dyl_function_slope as dyl
import imageio
import os
from numpy import ma
import cmaps

equence_font={
    'style': "Arial",
    'weight': "bold",
    'fontsize':7
}
plt.rcParams['font.family'] = 'Arial'
#plt.rcParams['figure.figsize'] = (2, 2)
ccmap = cmaps.ncl_default

#########################以下：计算meri 图数据#############################
ERA5 = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/ERA5/ua/ERA5_u_1958-2020_288x145_19levels_zonmean.nc').u[22:63,::-1,:,0]
JRA55 = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/JRA-55/ua/JRA55_u_1958-2022_288x145_19levels_zonmean.nc').UGRD_GDS0_ISBL_S123[22:63,::-1,:,0]
MERRA2 = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/MERRA-2/ua/u_MERRA2_1980-2022_zonmean_288x145_19levels_yearmean.nc').U[:41,:,:,0]
obs_data_U = [ERA5,MERRA2,JRA55]
obs_mean_trend_U, obs_mean_trend_p_values_U = dyl.calculate_trend_3D_ndarray(np.nanmean(obs_data_U,axis=0))
obs_mean_clim_U = np.nanmean(np.nanmean(obs_data_U,axis=0),axis=0)
mask_obs_mean_trend_U = obs_mean_trend_p_values_U < 0.05
significant_points_U = np.where(mask_obs_mean_trend_U, True, False)
##################T###################
ERA5_T = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/ERA5/ta/ta_ERA5_1979-2022_zonmean.nc').t[1:42,::-1,:,0]
JRA55_T = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/JRA-55/ta/ta_jra55_1979_2023_zonmean.nc').TMP_GDS0_ISBL[1:42,::-1,::-1,0]
MERRA2_T = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/MERRA-2/ta/ta_MERRA2_1980-2022_zonmean_delete2010.nc').T[:41,:,:,0]
#obs_data_T = [ERA5_T,MERRA2_T,JRA55_T]
obs_data_T = [ERA5_T,JRA55_T]
obs_mean_trend_T, obs_mean_trend_p_values_T = dyl.calculate_trend_3D_ndarray(np.nanmean(obs_data_T,axis=0))
#obs_mean_trend_T, obs_mean_trend_p_values_T = dyl.calculate_trend_3D_ndarray(obs_data_T[1])
obs_mean_clim_T = np.nanmean(np.nanmean(obs_data_T,axis=0),axis=0)
#obs_mean_clim_T = np.nanmean(obs_data_T[1],axis=0)
mask_obs_mean_trend_T = obs_mean_trend_p_values_T < 0.05
significant_points_T = np.where(mask_obs_mean_trend_T, True, False)


era5_level = ERA5.level
era5_level_label = list(map(int, era5_level.data.tolist()))
era5_lat = ERA5.lat
level_change = np.arange(-.5, .6, 0.1)
level_clim_U = np.arange(20, 50, 10)
level_clim_without20 = [-10,0,10]
level_clim_T=np.arange(200, 300, 10)

fig = plt.figure(figsize=(9, 9), dpi=500)
axes=[[0.1, 0.55, 0.3, 0.4],[0.5, 0.55, 0.3, 0.4],[0.1, 0.05, 0.3, 0.4],[0.5, 0.05, 0.3, 0.4]]

def plot_meri_section(ax, trend,mean_clim,level_range,level_clim_slim,level_range_clim,significant_points,title,sequence):
    cycle_data, cycle_mon = add_cyclic_point(trend, coord=era5_lat)
    cycle_MON, cycle_LEVEL = np.meshgrid(cycle_mon, np.arange(era5_level.shape[0]))
    cycle_MON = cycle_MON.filled(np.nan)
    cycle_data = cycle_data.filled(np.nan)
    ax1 = ax
    c1 = ax1.contourf(cycle_MON, cycle_LEVEL, cycle_data, cmap=ccmap, levels=level_range, extend='both')
    cycle_clim, cycle_mon = add_cyclic_point(mean_clim, coord=era5_lat)
    c2 = ax1.contour(cycle_MON, cycle_LEVEL, cycle_clim, levels=level_clim_slim, colors='k', alpha=0.6,
                     linewidths=1)
    c3 = ax1.contour(cycle_MON, cycle_LEVEL, cycle_clim, levels=level_range_clim, colors='k', alpha=0.6,
                     linewidths=1)
    ax1.clabel(c3, inline=True, fontsize=7)

    cycle_dot, cycle_mon = add_cyclic_point(significant_points, coord=era5_lat)
    significance = np.ma.masked_where(cycle_dot == False, cycle_dot)
    c3 = ax1.contourf(cycle_MON, cycle_LEVEL, significance, colors='none', hatches=['////'])
    for j, collection in enumerate(c3.collections):  ############更改打点的颜色
        collection.set_edgecolor('grey')
    for collection in c3.collections:
        collection.set_linewidth(0)
    ax1.clabel(c2, inline=True, fontsize=7)
    ax1.set_xticks([-90, -60, -30, 0, 30, 60, 90])  # 指定要显示的经纬度
    ax1.xaxis.set_major_formatter(LatitudeFormatter())  # 刻度格式转换为经纬度样式
    ax1.yaxis.set_ticks(np.arange(era5_level.shape[0]), era5_level_label)  # 指定要显示的经纬度
    ax1.tick_params(axis='x', labelsize=11)  # 设置x轴刻度数字大小
    ax1.tick_params(axis='y', labelsize=11)  # 设置y轴刻度数字大小
    ax1.set_title(title, loc='right', fontsize=12)
    ax1.set_ylabel('hPa', fontsize=12)
    ax1.text(-0.2, 1.1, sequence, transform=ax1.transAxes, fontsize=22, fontweight='bold', va='top', ha='left')

    return c1

def plot_meri_section_withoutsig(ax,trend,mean_clim,level_range,level_clim_slim,level_range_clim,title,sequence):
    cycle_data, cycle_mon = add_cyclic_point(trend, coord=era5_lat)
    cycle_MON, cycle_LEVEL = np.meshgrid(cycle_mon, np.arange(era5_level.shape[0]))
    cycle_MON = cycle_MON.filled(np.nan)
    cycle_data = cycle_data.filled(np.nan)
    ax1 = ax
    c1 = ax1.contourf(cycle_MON, cycle_LEVEL, cycle_data, cmap=ccmap, levels=level_range, extend='both')
    cycle_clim, cycle_mon = add_cyclic_point(mean_clim, coord=era5_lat)
    c2 = ax1.contour(cycle_MON, cycle_LEVEL, cycle_clim, levels=level_clim_slim, colors='k', alpha=0.6,
                     linewidths=1)
    c3 = ax1.contour(cycle_MON, cycle_LEVEL, cycle_clim, levels=level_range_clim, colors='k', alpha=0.6,
                     linewidths=1)
    ax1.clabel(c3, inline=True, fontsize=7)
    ax1.clabel(c2, inline=True, fontsize=7)
    ax1.set_xticks([-90, -60, -30, 0, 30, 60, 90])  # 指定要显示的经纬度
    ax1.xaxis.set_major_formatter(LatitudeFormatter())  # 刻度格式转换为经纬度样式
    ax1.yaxis.set_ticks(np.arange(era5_level.shape[0]), era5_level_label)  # 指定要显示的经纬度
    ax1.tick_params(axis='x', labelsize=11)  # 设置x轴刻度数字大小
    ax1.tick_params(axis='y', labelsize=11)  # 设置y轴刻度数字大小
    ax1.set_title(title, loc='right', fontsize=12)
    ax1.set_ylabel('hPa', fontsize=12)
    ax1.text(-0.2, 1.1, sequence, transform=ax1.transAxes, fontsize=22, fontweight='bold', va='top', ha='left')

    '''
    cb = fig.colorbar(c1, orientation='vertical', shrink=0.85, pad=0.05, extend='both')
    cb.ax.yaxis.set_major_locator(MultipleLocator(0.2))
    cb.set_ticks(level_change)
    # ax1.tick_params(axis='y', labelsize=8)  # 设置y轴刻度数字大小
    cb.ax.tick_params(axis='y', which='major', direction='in', length=10, labelsize=12)
    # cb.ax.tick_params(which='minor', direction='in', length=5)
    cb.set_label(label=bar_label, fontsize=12)
    plt.subplots_adjust(wspace=0.5)
    #plt.savefig("/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/figures/ta_OBSmme_meri_sec_1980-2020.png", dpi=500)
    plt.show()
    '''
    return c1

plot_meri_section(fig.add_axes(axes[0]),obs_mean_trend_T*10,obs_mean_clim_T,np.arange(-.5, .6, 0.1),level_clim_T,
                  [0],
                  significant_points_T,
                  'OBS','a')

##########################################################################
#obs 提取的VST及其std
Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization

def calculate_beta(T, delta_T, p):
    # T_fine = interpolate_to_fine(T, p, p_fine)
    # delta_T_fine = interpolate_to_fine(delta_T, p, p_fine)
    # Calculate es, dT_dp, and dT_des as before
    dT_dp = np.gradient(T, p, axis=0)

    # Calculate beta for each time, level, and latitude
    def func(beta, delta_T, dT_dp, T, level):
        # p_adjusted = np.tile(p_fine, (181, 1)).T
        return delta_T - (beta - 1) * (p[level] * dT_dp - Rv * T ** 2 / Lv)

    # Solve for beta
    beta_init = 1.15
    beta = np.empty_like(T)
    for lat in range(T.shape[1]):
        # for level in range(p.size):
        beta[:, lat] = optimize.newton(func, beta_init, args=(delta_T[lat], dT_dp[5, lat], T[5, lat], 5))
    # Return the calculated beta array
    return beta


def calculate_transformed_u(u, beta, p):
    # Calculate u_prime as before using interpolated u and beta
    # Calculate transformed u'
    u_prime = np.empty_like(u)
    for lat in range(u.shape[1]):
        # for level in range(p.size):
        transformed_p = beta[:, lat] * p
        u_prime[::-1, lat] = np.interp(transformed_p[::-1], p[::-1], u[::-1, lat])
    # Return the calculated u_prime array
    return u_prime

p = ERA5_T.level
delta_T_hist = obs_mean_trend_T*80
beta_hist = calculate_beta(obs_mean_clim_T, delta_T_hist[5],p)
t_prime_hist = calculate_transformed_u(obs_mean_clim_T, beta_hist, p)
u_prime_hist = calculate_transformed_u(obs_mean_clim_U, beta_hist,p)
delta_VST_u245 = u_prime_hist - obs_mean_clim_U #保证delta_VST_u245为Dataarray,便于后续使用areamean.mask_am函数
#delta_VST_u245 = delta_VST_u245.expand_dims(dim='new_dim', axis=3)/80 ####恢复1年的向上抬升贡献
delta_VST_u245 = delta_VST_u245/80
delta_VST_t = (t_prime_hist - obs_mean_clim_T)/80

c1 = plot_meri_section_withoutsig(fig.add_axes(axes[1]),delta_VST_t*10,obs_mean_clim_T,np.arange(-.5, .6, 0.1),level_clim_T,
                 [0],
                 'VST(OBS)','b')

cbar_ax = fig.add_axes([0.87, 0.6, 0.01, 0.3])  # 调整这些数字以改变colorbar的大小和位置
cb = plt.colorbar(c1,cax=cbar_ax , orientation='vertical', shrink=0.85, pad=0.05, extend='both')
cb.ax.yaxis.set_major_locator(MultipleLocator(0.2))
cb.set_ticks(level_change)
# ax1.tick_params(axis='y', labelsize=8)  # 设置y轴刻度数字大小
cb.ax.tick_params(axis='y', which='major', direction='in', length=3, labelsize=12)
# cb.ax.tick_params(which='minor', direction='in', length=5)
cb.set_label(label='$^\circ$C decade$^{-1}$', fontsize=12)
plt.subplots_adjust(wspace=0.5)
##########################################################################
plot_meri_section_withoutsig(fig.add_axes(axes[3]),delta_VST_u245*10,obs_mean_clim_U,np.arange(-.5, .6, 0.1),level_clim_without20,level_clim_U,
                  'VST(OBS)','d')
c2 = plot_meri_section(fig.add_axes(axes[2]),obs_mean_trend_U*10,obs_mean_clim_U,np.arange(-.5, .6, 0.1),level_clim_U,level_clim_without20,
                  significant_points_U,
                  'OBS','c')

cbar_ax = fig.add_axes([0.85, 0.1, 0.01, 0.3])  # 调整这些数字以改变colorbar的大小和位置
cb = plt.colorbar(c2,cax=cbar_ax , orientation='vertical', shrink=0.85, pad=0.05, extend='both')
cb.ax.yaxis.set_major_locator(MultipleLocator(0.2))
cb.set_ticks(level_change)
# ax1.tick_params(axis='y', labelsize=8)  # 设置y轴刻度数字大小
cb.ax.tick_params(axis='y', which='major', direction='in', length=3, labelsize=12)
# cb.ax.tick_params(which='minor', direction='in', length=5)
cb.set_label(label='m s$^{-1}$ decade$^{-1}$', fontsize=12)
plt.subplots_adjust(wspace=0.5)

plt.savefig("/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/figures/formal_work/ta_ua_OBS_VST_meri_sec_1980-2020.png", dpi=500)
plt.show()
