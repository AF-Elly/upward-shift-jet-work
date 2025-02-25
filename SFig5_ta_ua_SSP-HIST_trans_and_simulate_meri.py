import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.patches as mpatches
import function_beta_VSTtrans as VST
import cmaps
sequence_font={
    'style': "Arial",
    'weight': "bold",
    'fontsize':7
}
plt.rcParams['font.family'] = 'Arial'
ccmap = cmaps.ncl_default

T_hist585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_1980-2010_zonmean_historical_289x145.nc').ta[
            :, 0, :, :, 0]
T_future585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_2060-2090_zonmean_future_289x145.nc').ta[
              :, 0, :, :, 0]
ua_hist585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_1980-2010_zonmean_historical_289x145.nc').ua[
             :, 0, :, :, 0]
ua_future585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_2060-2090_zonmean_future_289x145.nc').ua[
               :, 0, :, :, 0]

delta_T585 = T_future585 - T_hist585
# Constants
Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization
p = T_hist585.plev  # Pressure levels
g=9.8

level = T_hist585.plev / 100
level_label = list(map(int, level.data.tolist()))

beta585 = np.zeros(T_hist585.shape)
t_prime585 = np.zeros(T_hist585.shape)
u_prime585 = np.zeros(ua_hist585.shape)
ua_prime585 = np.zeros(ua_hist585.shape)
# Main script execution
for models in range(T_hist585.shape[0]):
    # T = np.mean(T_historical,0)
    # delta_T = np.mean(delta_T,0)
    beta585[models] = VST.calculate_beta(T_hist585[models], delta_T585[models], p)
    t_prime585[models] = VST.calculate_transformed_t(T_hist585[models], beta585[models], p)
    ua_prime585[models] = VST.calculate_transformed_q(ua_hist585[models], beta585[models], p)
    # u_prime585[models] = VST.calculate_transformed_u(u_hist585[models], beta585[models], p)

delta_VST_t585 = t_prime585 - T_hist585
delta_t585 = T_future585 - T_hist585
delta_VST_u585 = ua_prime585 - ua_hist585
delta_u585 = ua_future585 - ua_hist585
delta_VST_t585 = delta_VST_t585.expand_dims(dim='new_dim', axis=3)
delta_t585 = delta_t585.expand_dims(dim='new_dim', axis=3)
delta_VST_u585 = delta_VST_u585.expand_dims(dim='new_dim', axis=3)
delta_u585 = delta_u585.expand_dims(dim='new_dim', axis=3)
#ua_hist = ua_hist585.expand_dims(dim='new_dim', axis=3)


###p_level:array([100000.,  92500.,  85000.,  70000.,  60000.,  50000.,  40000.,  30000.,
###        25000.,  20000.,  15000.,  10000.,   7000.,   5000.,   3000.,   2000.,
###         1000.,    500.,    100.])
p_lower_lev_label = 9
p_upper_lev_label = 11

# 模拟数据
np.random.seed(0)
# 使用 28 个不同的符号表示不同的模式
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '1', 'h', 'H', 'D', 'd', 'P', 'X']
models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'CanESM5', 'CAS-ESM2-0', 'CESM2-WACCM',
          'CMCC-CM2-SR5',
          'CMCC-ESM2', 'EC-Earth3', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0',
          'GFDL-ESM4', 'IITM-ESM', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR',
          'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1']

level = T_hist585.plev / 100
level_label = list(map(int, level.data.tolist()))
lat = T_hist585.lat
level = np.arange(-5, 6, 1)
plot_data_all_models = [delta_t585[:,:, :,0],delta_VST_t585[:, :, :,0],
                        delta_u585[:,:, :,0],delta_VST_u585[:, :, :,0]]
plot_data=[np.nanmean(delta_t585[:, :, :, 0], axis=0),
           np.nanmean(delta_VST_t585[:, :, :, 0], axis=0),
           np.nanmean(delta_u585[:,:, :,0],axis=0),
           np.nanmean(delta_VST_u585[:, :, :,0],axis=0)]

####################检验模式间一致性#############
threshold = 0.8  # 80% condition
n_models = delta_u585.shape[0]  # Total number of models (28)
consistency = []
for i in range(4):
    # Get the sign of the ensemble mean slope (h_ssp585_slope_mme)
    mean_sign = np.sign(plot_data_all_models[i])  # Shape: (19, 145)
    # Get the sign of all models' slopes
    model_signs = np.sign(plot_data[i])  # Shape: (28, 19, 145)
    # Compare each model's sign with the ensemble mean sign
    same_sign_count = np.sum(model_signs == mean_sign, axis=0)  # Shape: (19, 145)
    # Determine significant regions where >=80% of models have the same sign as the ensemble mean
    significance_mask = (same_sign_count / n_models) >= threshold  # Shape: (19, 145)
    significant_points = np.where(significance_mask, True, False)
    consistency.append(significant_points)

plot_data_title = ['SSP585-HIST','VST(SSP585-HIST)','SSP585-HIST','VST(SSP585-HIST)']
sub_add_axes = [[0.1, 0.15, 0.2, 0.7], [0.4, 0.15, 0.2, 0.7],[0.7, 0.15, 0.2, 0.7]]
sequence = ['a','b','c','d']
fig = plt.figure(figsize=(9, 9), dpi=500)
axes=[[0.1, 0.55, 0.3, 0.4],[0.5, 0.55, 0.3, 0.4],[0.1, 0.05, 0.3, 0.4],[0.5, 0.05, 0.3, 0.4]]

#for i in range(4):
def plot_meri_section(ax, trend, mean_clim, level_range, level_clim_slim, level_range_clim, significant_points,
                      title, sequence):
    ax = ax
    cycle_data, cycle_lat = add_cyclic_point(trend, coord=lat)  # delta_u585[i * 7 + j][:, :, 0]-
    cycle_clim, cycle_lat = add_cyclic_point(mean_clim, coord=lat)
    cycle_LAT, cycle_LEVEL = np.meshgrid(cycle_lat, np.arange(19))
    cycle_LAT = cycle_LAT.filled(np.nan)
    cycle_data = cycle_data.filled(np.nan)

    cc = ax.contourf(cycle_LAT, cycle_LEVEL, cycle_data, levels=level_range, cmap=ccmap, extend='both')
    c2 = ax.contour(cycle_LAT, cycle_LEVEL, cycle_clim, levels=level_clim_slim, colors='k', alpha=0.6,
                     linewidths=1)
    c3 = ax.contour(cycle_LAT, cycle_LEVEL, cycle_clim, levels=level_range_clim, colors='k', alpha=0.6,
                     linewidths=1)
    ax.clabel(c3, inline=True, fontsize=7)
    ax.clabel(c2, inline=True, fontsize=7)

    cycle_dot, cycle_mon = add_cyclic_point(significant_points, coord=lat)
    significance = np.ma.masked_where(cycle_dot == False, cycle_dot)
    c4 = ax.contourf(cycle_LAT, cycle_LEVEL, significance, colors='none', hatches=['////'])
    for j, collection in enumerate(c4.collections):  ############更改打点的颜色
        collection.set_edgecolor('silver')
    for collection in c4.collections:
        collection.set_linewidth(0)

    ax.yaxis.set_ticks(np.arange(19), level_label)  # 指定要显示的经纬度
    ax.tick_params(axis='y', labelsize=11)  # 设置y轴刻度数字大小
    #cbar_ax = fig.add_axes([0.91, 0.15, 0.01, 0.3])  # 调整这些数字以改变colorbar的大小和位置
    #cbar = plt.colorbar(cc, cax=cbar_ax, orientation='vertical', shrink=0.85, pad=0.05, extend='both')
    #cbar.set_ticks(level)
    #cbar.ax.tick_params(axis='both', which='major', direction='in', length=7.5, labelsize=14)
    #cbar.set_label(label='u (m/s)', fontsize=14)

    ax.set_xticks([-90, -60, -30, 0, 30, 60, 90])  # 指定要显示的经纬度
    ax.xaxis.set_major_formatter(LatitudeFormatter())  # 刻度格式转换为经纬度样式
    ax.tick_params(axis='x', labelsize=11)  # 设置x轴刻度数字大小
    '''
    ax.add_patch(mpatches.Rectangle((-40, 8), 20, 2,
                       fill=False,
                       color="k",
                       linewidth=1.5))
    ax.add_patch(mpatches.Rectangle((20, 8), 20, 2,
                                    fill=False,
                                    color="k",
                                    linewidth=1.5))
    '''
    ax.set_title(title, loc='right', fontsize=12)
    ax.set_ylabel('hPa', fontsize=12)
    ax.text(-0.2, 1.1, sequence, transform=ax.transAxes, fontsize=22, fontweight='bold', va='top', ha='left')
    return cc

plot_meri_section(fig.add_axes(axes[0]),plot_data[0],mean_clim=np.nanmean(T_hist585,0),
                  level_range=np.arange(-5, 5.1, 1),level_clim_slim=np.arange(200,300,10),level_range_clim=np.arange(200,300,10),
                  significant_points=consistency[0],
                  title=plot_data_title[0],sequence=sequence[0])
cc=plot_meri_section(fig.add_axes(axes[1]),plot_data[1],mean_clim=np.nanmean(T_hist585,0),
                  level_range=np.arange(-5, 5.1, 1),level_clim_slim=np.arange(200,300,10),level_range_clim=np.arange(200,300,10),
                  significant_points=consistency[1],
                  title=plot_data_title[1],sequence=sequence[1])

cbar_ax = fig.add_axes([0.87, 0.6, 0.01, 0.3])  # 调整这些数字以改变colorbar的大小和位置
cbar = plt.colorbar(cc, cax=cbar_ax, orientation='vertical', shrink=0.85, pad=0.05, extend='both')
cbar.set_ticks(np.arange(-5, 5.1, 1))
cbar.ax.tick_params(axis='both', which='major', direction='in', length=4, labelsize=12)
cbar.set_label(label='$^\circ$C', fontsize=12)

plot_meri_section(fig.add_axes(axes[2]),plot_data[2],mean_clim=np.nanmean(ua_hist585,0),
                  level_range=np.arange(-5, 5.1, 1),level_clim_slim=np.arange(-20,40,10),level_range_clim=np.arange(-20,40,10),
                  significant_points=consistency[2],
                  title=plot_data_title[2],sequence=sequence[2])
cc=plot_meri_section(fig.add_axes(axes[3]),plot_data[3],mean_clim=np.nanmean(ua_hist585,0),
                  level_range=np.arange(-5, 5.1, 1),level_clim_slim=np.arange(-20,40,10),level_range_clim=np.arange(-20,40,10),
                  significant_points=consistency[3],
                  title=plot_data_title[3],sequence=sequence[3])

cbar_ax = fig.add_axes([0.85, 0.1, 0.01, 0.3])  # 调整这些数字以改变colorbar的大小和位置
cbar = plt.colorbar(cc, cax=cbar_ax, orientation='vertical', shrink=0.85, pad=0.05, extend='both')
cbar.set_ticks(np.arange(-5, 5.1, 1))
cbar.ax.tick_params(axis='both', which='major', direction='in', length=3, labelsize=12)
cbar.set_label(label='m s$^{-1}$', fontsize=12)

plt.savefig('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/figures/formal_work/ta_ua_SSP-HIST_VST_meri_sec.png',dpi=600)
plt.show()