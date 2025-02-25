import numpy as np
import xarray as xr
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
from cartopy.util import add_cyclic_point
from scipy.stats import linregress
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl
import cmaps

plt.rcParams['font.family'] = 'Arial'
ccmap = cmaps.cmp_b2r
ssp245 = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ua_h+ssp245_all_models_197901-209912_289x145_100_200hPa_1980-2010_vs_2060-2090.nc').ua[:,0,:,:,:]
ssp585 = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_197901-209912_289x145_100_200hPa_1980-2010_vs_2060-2090.nc').ua[:,0,:,:,:]
climatology_historical = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/all_models_climatology_1980-2010.nc').ua[:,0,:,:,:]
ssp245_zonmean = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ua_h+ssp245_all_models_197901-209912_289x145_100_200hPa_1980-2010_vs_2060-2090_zonmean.nc').ua[:,0,:,:,0]
ssp585_zonmean = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_197901-209912_289x145_100_200hPa_1980-2010_vs_2060-2090_zonmean.nc').ua[:,0,:,:,0]
def calculate_consistency(model_trends, ensemble_trend):
    num_models = model_trends.shape[0]
    consistency_mask = np.full(ensemble_trend.shape, False)
    for i in range(ensemble_trend.shape[0]):
        for j in range(ensemble_trend.shape[1]):
            # 计算与集合平均趋势符号一致的模式数量
            num_consistent_models = np.sum(np.sign(model_trends[:, i, j]) == np.sign(ensemble_trend[i, j]))
            # 判断是否超过80%
            if num_consistent_models / num_models >= 0.8:
                consistency_mask[i, j] = True
    return consistency_mask

#ssp245_100change=np.empty_like(np.mean(ssp245_100minus200,1))
#ssp585_model_change=np.empty_like(np.mean(ssp585_100minus200,1))

# 计算集合平均趋势
ssp245_100change_mme = np.nanmean(ssp245[:,1],0)
ssp245_200change_mme = np.nanmean(ssp245[:,0],0)
ssp245_100minus200change_mme = np.nanmean(ssp245[:,1]-ssp245[:,0],0)
ssp585_100change_mme = np.nanmean(ssp585[:,1],0)
ssp585_200change_mme = np.nanmean(ssp585[:,0],0)
ssp585_100minus200change_mme = np.nanmean(ssp585[:,1]-ssp585[:,0],0)
climatology_mme = np.nanmean(climatology_historical,0)
climatology_zonmean = np.nanmean(climatology_historical,3)
# 计算模式间一致性
ssp245_consistency_mask_100 = calculate_consistency(np.array(ssp245[:,1]), np.array(ssp245_100change_mme))
ssp245_consistency_mask_200 = calculate_consistency(np.array(ssp245[:,0]), np.array(ssp245_200change_mme))
ssp245_consistency_mask_100minus200 = calculate_consistency(np.array(ssp245[:,1]-ssp245[:,0]), np.array(ssp245_100minus200change_mme))
ssp585_consistency_mask_100 = calculate_consistency(np.array(ssp585[:,1]), np.array(ssp585_100change_mme))
ssp585_consistency_mask_200 = calculate_consistency(np.array(ssp585[:,0]), np.array(ssp585_200change_mme))
ssp585_consistency_mask_100minus200 = calculate_consistency(np.array(ssp585[:,1]-ssp585[:,0]), np.array(ssp585_100minus200change_mme))

# 绘制子图
fig = plt.figure(figsize=(16, 6),dpi=300)
leftlon, rightlon, lowerlat, upperlat = (-180, 181, -90, 90)
# 绘制趋势图并添加显著性阴影
def plot_trend_and_significance(subloc,climatology, data, trend, consistency_mask, title,sequence, exp, levels, levels2, ccmap):
    # Set the global extent
    #ax.projection(ccrs.Robinson())
    ax = fig.add_subplot(subloc, projection=ccrs.Robinson())
    ax.set_global()
    ax.set_aspect(1.25)
    # 为数据添加周期性点
    trend_cyclic, lon_cyclic = add_cyclic_point(trend, coord=data.lon)

    # 确保一致性掩码也添加周期性点
    consistency_mask_cyclic, _ = add_cyclic_point(consistency_mask, coord=data.lon)

    # Plot the trend
    cf = ax.contourf(lon_cyclic, data.lat, trend_cyclic, levels=levels, cmap=ccmap, extend='both', transform=ccrs.PlateCarree())
    #ax.coastlines()
    ax.add_feature(cfeature.COASTLINE.with_scale('10m'), lw=0.25)
    # Add gridlines and labels
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    #ax.tick_params(axis='y', labelsize=10)
    #ax.tick_params(axis='x', labelsize=10)

    # 在plot_trend_and_significance函数内部，在绘制显著性的代码块之后，添加以下代码来绘制一致性阴影
    consistency = np.ma.masked_where(~consistency_mask_cyclic, trend_cyclic)
    cc=ax.contourf(lon_cyclic, data.lat, consistency, hatches=['...'], colors='none', transform=ccrs.PlateCarree())
    for j, collection in enumerate(cc.collections):  ############更改打点的颜色
        collection.set_edgecolor('silver')
    for collection in cc.collections:
        collection.set_linewidth(0)
    clim_cyclic, _ = add_cyclic_point(climatology, coord=data.lon)
    c2 = ax.contour(lon_cyclic, data.lat, clim_cyclic, levels=levels2, colors='snow', alpha=1, linewidths=1,transform=ccrs.PlateCarree())
    ax.clabel(c2,colors='k', inline=True, fontsize=7)
    # Set the title
    ax.set_title(title, fontsize=14)
    ax.text(-0.1, 1.2, sequence, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')
    ax.text(1.05, 1.1, exp, transform=ax.transAxes, fontsize=12, va='top', ha='right')
    return cf

levels = np.linspace(-5, 5, 11)
levels2 = np.linspace(-20,40,7)
# SSP2-4.5子图
cf11 = plot_trend_and_significance(241, climatology_mme[1], ssp245[:,1], ssp245_100change_mme, ssp245_consistency_mask_100, 'u$_{100}$', 'a', 'SSP245', levels, levels2, ccmap)
cf12 = plot_trend_and_significance(242, climatology_mme[0], ssp245[:,0], ssp245_200change_mme, ssp245_consistency_mask_200, 'u$_{200}$', 'b', 'SSP245',levels,levels2,ccmap)
cf13 = plot_trend_and_significance(243, climatology_mme[1]-climatology_mme[0], ssp245[:,1]-ssp245[:,0], ssp245_100minus200change_mme, ssp245_consistency_mask_100minus200,
                                   'u$_{100}$ - u$_{200}$','c','SSP245',levels,levels2,ccmap)

# SSP5-8.5子图
cf21 = plot_trend_and_significance(245, climatology_mme[1], ssp585[:,1], ssp585_100change_mme, ssp585_consistency_mask_100, 'u$_{100}$', 'e','SSP585', levels,levels2,ccmap)
cf22 = plot_trend_and_significance(246, climatology_mme[0], ssp585[:,0], ssp585_200change_mme, ssp585_consistency_mask_200, 'u$_{200}$', 'f', 'SSP585',levels, levels2, ccmap)
cf23 = plot_trend_and_significance(247, climatology_mme[1]-climatology_mme[0], ssp585[:,1]-ssp585[:,0], ssp585_100minus200change_mme, ssp585_consistency_mask_100minus200,
                                   'u$_{100}$ - u$_{200}$', 'g','SSP585', levels,levels2, ccmap)

# 在SSP2-4.5子图右上角添加文字
#ax[0].text(0.95, 1.15, 'SSP2-4.5', transform=ax[0].transAxes, fontsize=15, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='none', alpha=0))
# 在SSP5-8.5子图右上角添加文字
#ax[1].text(0.95, 1.15, 'SSP5-8.5', transform=ax[1].transAxes, fontsize=15, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='none', alpha=0))

# 绘制SSP245和SSP585的纬向风差值子图，并调整宽度和标签
def plot_zonmean_diff(ax, data100, data200,mmecolor, shadecolor, title, sequence, xlim, lat_data=ssp245_zonmean[:, 1, :]):
    # 计算100hPa和200hPa的纬向风差值
    diff = data100 - data200
    # 计算集合平均
    #mean_diff = diff.mean(axis=0,skipna=True)
    mean_diff = np.nanmean(diff,axis=0)
    # 计算标准差
    std_diff = np.nanstd(diff,axis=0)
    # 获取纬度值
    latitudes = lat_data.lat.values
    # 绘制集合平均
    ax.plot(mean_diff, latitudes, color=mmecolor, label='MME')
    # 填充模型间一倍标准差
    ax.fill_betweenx(latitudes, mean_diff - std_diff, mean_diff + std_diff, color=shadecolor, alpha=0.1)
    ax.fill_between(xlim, -20, -40, color='dodgerblue', alpha=0.05)
    ax.fill_between(xlim, 20, 40, color='dodgerblue', alpha=0.05)
    # 绘制竖直0刻度线
    ax.axvline(0, color='grey', linestyle='-', linewidth=0.5)
    # 绘制横向30刻度线
    #ax.axhline(30, color='steelblue', linestyle=':', linewidth=2)
    #ax.axhline(-30, color='steelblue', linestyle=':', linewidth=2)
    # 设置横纵坐标范围
    ax.set_xlim(xlim)
    ax.set_ylim([-90, 90])
    ax.xaxis.set_major_locator(MultipleLocator(2))
    # 设置纬度标签
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_yticklabels(['90°S', '60°S', '30°S', '0°', '30°N', '60°N', '90°N'])
    ax.tick_params(axis='y',colors='k', labelsize=10, pad=2,length=4, width=1,labelcolor='k')
    ax.tick_params(axis='x',colors=mmecolor, labelsize=10, pad=2,length=4, width=1,labelcolor=mmecolor)
    #ax2.tick_params(axis='y', colors=color[i], pad=6, direction='out', length=4, width=1,
    #                labelsize=12, labelcolor=color[i])
    #ax.set_xticks(np.linspace(-0.2, 0.6, .1))
    # 设置标题
    #ax.set_title(title, fontsize=14)
    # 设置坐标轴标签
    #ax.set_xlabel('Zonal Wind Difference (m/s)')
    ax.set_ylabel('Latitude',fontsize=14)
    ax.text(-0.6, 1.2, sequence, transform=ax.transAxes, fontsize=20, fontweight='bold', va='top', ha='left')
# 添加子图并调整大小
ax14 = fig.add_axes([0.765, 0.57, 0.07, 0.3])  # 调整这些数字以改变大小和位置
ax14_ = ax14.twiny()
ax24 = fig.add_axes([0.765, 0.14, 0.07, 0.3])  # 调整这些数字以改变大小和位置
ax24_ = ax24.twiny()
ax24.set_xlabel('m s$^{-1}$',fontsize=14)
ax14_.set_xlabel('m s$^{-1}$', fontsize=14)
# 绘制子图
plot_zonmean_diff(ax14, ssp245_zonmean[:, 1, :], ssp245_zonmean[:, 0, :], 'orange','orange', 'u$_{100}$ - u$_{200}$', 'd',(-3,5))
ax14.set_xticks(np.linspace(-2, 4, 4))
plot_zonmean_diff(ax14_, climatology_zonmean[:,1,:], climatology_zonmean[:, 0, :],'k','k', ' ', ' ',(-30,50))
ax14_.set_xticks(np.linspace(-20, 40, 4))
plot_zonmean_diff(ax24, ssp585_zonmean[:, 1, :], ssp585_zonmean[:, 0, :], 'r','r', 'u$_{100}$ - u$_{200}$', 'h',(-3,5))
ax24.set_xticks(np.linspace(-2, 4, 4))
plot_zonmean_diff(ax24_, climatology_zonmean[:,1,:], climatology_zonmean[:, 0, :],'k','k', ' ', ' ',(-30,50))
ax24_.set_xticks(np.linspace(-20, 40, 4))
# 调整布局
#plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# 创建一个大的轴，然后将colorbar放在下方
cbar_ax = fig.add_axes([0.2, 0.08, 0.4, 0.02])  # 调整这些数字以改变colorbar的大小和位置
# 创建colorbar
cbar = plt.colorbar(cf11, cax=cbar_ax, orientation='horizontal',pad=0.08, aspect=50, shrink=0.7, extend='both')
cbar.ax.xaxis.set_major_locator(MultipleLocator(0.2))
cbar.set_ticks(levels)
cbar.ax.tick_params(which='major', direction='in', length=7.5)
cbar.set_label(label='u change (m s$^{-1}$)', fontsize=14,labelpad=0.09)
plt.savefig('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/figures/formal_work/ssp_global_pattern_change.png',dpi=600)
# 显示图形
plt.show()




