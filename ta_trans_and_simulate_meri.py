import numpy as np
import xarray as xr
from scipy import optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from scipy import stats
import areamean_dhq as areamean
from matplotlib.font_manager import FontProperties
import cmaps
ccmap = cmaps.ncl_default
plt.rcParams['font.family'] = 'Arial'

T_hist245 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ta_h+ssp245_all_models_1980-2010_zonmean_historical_289x145.nc').hus[
            :, 0, :, :, 0]
T_future245 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ta_h+ssp245_all_models_2060-2090_zonmean_future_289x145.nc').hus[
              :, 0, :, :, 0]
T_hist585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_1980-2010_zonmean_historical_289x145.nc').ta[
            :, 0, :, :, 0]
T_future585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_2060-2090_zonmean_future_289x145.nc').ta[
              :, 0, :, :, 0]
delta_T245 = T_future245 - T_hist585
delta_T585 = T_future585 - T_hist585
# Constants
Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization
p = T_hist585.plev  # Pressure levels


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
        beta[:, lat] = optimize.newton(func, beta_init, args=(delta_T[5, lat], dT_dp[5, lat], T[5, lat], 5))
    # Return the calculated beta array
    return beta

#T ′(p) = T (βp) − (β−1/β)* R v/ L *T(βp)**2.
def calculate_transformed_t(t, beta, p):
    # Calculate u_prime as before using interpolated u and beta
    # Calculate transformed u'
    t_betap = np.empty_like(t)
    t_prime = np.empty_like(t)
    for lat in range(t.shape[1]):
        # for level in range(p.size):
        transformed_p = beta[:, lat] * p
        t_betap[::-1, lat] = np.interp(transformed_p[::-1], p[::-1], t[::-1, lat])
    t_prime = t_betap-(beta-1)*Rv*t_betap**2/(beta*Lv)
    # Return the calculated u_prime array
    return t_prime


level = T_hist585.plev / 100
level_label = list(map(int, level.data.tolist()))
beta245 = np.zeros(T_hist585.shape)
t_prime245 = np.zeros(T_hist585.shape)
beta585 = np.zeros(T_hist585.shape)
t_prime585 = np.zeros(T_hist585.shape)
# Main script execution
for models in range(T_hist585.shape[0]):
    # T = np.mean(T_historical,0)
    # delta_T = np.mean(delta_T,0)
    beta245[models] = calculate_beta(T_hist585[models], delta_T245[models], p)
    t_prime245[models] = calculate_transformed_t(T_hist245[models], beta245[models], p)
    beta585[models] = calculate_beta(T_hist585[models], delta_T585[models], p)
    t_prime585[models] = calculate_transformed_t(T_hist585[models], beta585[models], p)
delta_VST_u245 = t_prime245 - T_hist245
delta_u245 = T_future245 - T_hist245
#delta_VST_u245 = delta_VST_u245.expand_dims(dim='new_dim', axis=3)
#delta_u245 = delta_u245.expand_dims(dim='new_dim', axis=3)
delta_VST_u585 = t_prime585 - T_hist585
delta_u585 = T_future585 - T_hist585
#delta_VST_u585 = delta_VST_u585.expand_dims(dim='new_dim', axis=3)
#delta_u585 = delta_u585.expand_dims(dim='new_dim', axis=3)


###p_level:array([100000.,  92500.,  85000.,  70000.,  60000.,  50000.,  40000.,  30000.,
###        25000.,  20000.,  15000.,  10000.,   7000.,   5000.,   3000.,   2000.,
###         1000.,    500.,    100.])
p_lower_lev_label=5
p_upper_lev_label=7

delta_VST_u_100_200_mme = [np.nanmean(delta_VST_u585,0),np.nanmean(delta_u585,0)]

level = T_hist585.plev / 100
level_label = list(map(int, level.data.tolist()))
lat = T_hist585.lat
level_change = np.arange(-10, 10.1, 1)
level_clim_without20 = np.arange(200, 301, 10)
sub_add_axes = [[0.11, 0.15, 0.23, 0.72], [0.39, 0.15, 0.23, 0.72],[0.67, 0.15, 0.23, 0.72]]
fig = plt.figure(figsize=(12, 5), dpi=400)
for i in range(2):
    ax1 = fig.add_axes(sub_add_axes[i])
    cycle_data, cycle_mon = add_cyclic_point(delta_VST_u_100_200_mme[i], coord=lat)
    cycle_MON, cycle_LEVEL = np.meshgrid(cycle_mon, np.arange(19))
    cycle_MON = cycle_MON.filled(np.nan)
    cycle_data = cycle_data.filled(np.nan)

    c1 = ax1.contourf(cycle_MON, cycle_LEVEL, cycle_data, cmap=ccmap, levels=level_change, extend='both')
    cycle_clim, cycle_mon = add_cyclic_point(np.nanmean(T_hist585,0), coord=lat)
    c2 = ax1.contour(cycle_MON, cycle_LEVEL, cycle_clim, levels=level_clim_without20, colors='k', alpha=0.6,
                     linewidths=0.8)
    #cycle_dot, cycle_mon = add_cyclic_point(subplot_dot[i, :, :], coord=lat)
    #c3 = ax1.contour(cycle_MON, cycle_LEVEL, cycle_clim, levels=[20, 25, 30, 35, 40], colors='k', alpha=0.6,
    #                 linewidths=1.5)
    ax1.clabel(c2, inline=True, fontsize=7)
    #c3 = ax1.contourf(cycle_MON, cycle_LEVEL, cycle_dot, levels=[0, 7.2, 9], colors='none', hatches=[None, '////'])
    '''    
    for j, collection in enumerate(c3.collections):  ############更改打点的颜色
        collection.set_edgecolor('silver')
    for collection in c3.collections:
        collection.set_linewidth(0)
    ax1.text(-0.15, 1.13, sequence[i], transform=ax1.transAxes, fontsize=24, fontweight='bold', va='top', ha='left')
    '''
    if i==0:
        ax1.clabel(c2, inline=True, fontsize=7)
        ax1.set_xticks([-90, -60, -30, 0, 30, 60, 90])  # 指定要显示的经纬度
        ax1.xaxis.set_major_formatter(LatitudeFormatter())  # 刻度格式转换为经纬度样式
        ax1.yaxis.set_ticks(np.arange(19), level_label)  # 指定要显示的经纬度
        ax1.tick_params(axis='x', labelsize=10)  # 设置x轴刻度数字大小
        ax1.tick_params(axis='y', labelsize=10)  # 设置y轴刻度数字大小
        #ax1.set_title(models[i], loc='left', fontsize=12)
        ax1.set_ylabel('Level (hPa)', fontsize=13)
        # ax1.axvline(x=0, ymin=0, ymax=12, linestyle='--', linewidth=2, color='b')
    else:
        ax1.clabel(c2, inline=True, fontsize=7)
        ax1.set_xticks([-90, -60, -30, 0, 30, 60, 90])  # 指定要显示的经纬度
        ax1.xaxis.set_major_formatter(LatitudeFormatter())  # 刻度格式转换为经纬度样式
        ax1.tick_params(axis='x', labelsize=10)  # 设置x轴刻度数字大小
        ax1.yaxis.set_ticks(np.arange(19), [])
        #ax1.set_title(models[i], loc='left', fontsize=12)


cbar_ax = fig.add_axes([0.92, 0.2, 0.01, 0.6])  # 调整这些数字以改变colorbar的大小和位置
# 创建colorbar
cb = plt.colorbar(c1, cax=cbar_ax, orientation='vertical',shrink=0.85, pad=0.05, extend='both')
    # if i==0:
cb.ax.yaxis.set_major_locator(MultipleLocator(1))
    # cb.ax.yaxis.set_minor_locator(AutoMinorLocator(4))
    # else:
    #    cb.ax.yaxis.set_major_locator(MultipleLocator(0.5))
    # cb.ax.yaxis.set_minor_locator(AutoMinorLocator(4))
cb.set_ticks(level_change)
cb.ax.tick_params(which='major', direction='in', length=10, labelsize=10)
    # cb.ax.tick_params(which='minor', direction='in', length=5)
cb.set_label(label='temp change ($^\circ$C)', fontsize=13)
cb.ax.tick_params(labelsize=12)
    #ax1.invert_yaxis()
plt.show()