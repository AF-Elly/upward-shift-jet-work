import numpy as np
import xarray as xr
from scipy import optimize
import copy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from scipy import stats
from matplotlib.colors import BoundaryNorm
from cartopy.util import add_cyclic_point
from scipy.stats import linregress

# 绘制观测趋势图并添加显著性阴影
def plot_trend_and_significance(axes, data, trend, p_values, title, sequence, exp, levels, levels2, ccmap, climatology=None,hzl=False):
    # Set the global extent
    # ax.projection(ccrs.Robinson())
    time, lat, lon = data.dims
    ax = fig.add_axes(axes, projection=ccrs.Robinson())
    ax.set_global()
    ax.set_aspect(1.25)

    if hzl:
        # ======= 添加20-40°N紫色横线 =======
        for lats in [20, 40]:
            ax.plot([-180, 180], [lats, lats], color='magenta', linewidth=1.2, linestyle='-',
                    transform=ccrs.PlateCarree(), zorder=4,alpha=0.8)

    # 为数据添加周期性点
    trend_cyclic, lon_cyclic = add_cyclic_point(trend, coord=data[lon])
    # 确保一致性掩码也添加周期性点
    consistency_mask_cyclic, _ = add_cyclic_point(p_values, coord=data[lon])

    # Plot the trend
    cf = ax.contourf(lon_cyclic, data[lat], trend_cyclic, levels=levels, cmap=ccmap, extend='both',
                     transform=ccrs.PlateCarree())
    ax.coastlines(lw=0.5)

    # Add gridlines and labels
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}

    # Plot the significance
    significance = np.ma.masked_where(p_values >= 0.05, p_values)
    cc = ax.contourf(data[lon], data[lat], significance, hatches=['...'], colors='none', transform=ccrs.PlateCarree())
    if hasattr(cc, 'collections'):
        collections = cc.collections
    else:
        collections = [cc]
    # 设置 hatch 样式
    for collection in collections:
        collection.set_edgecolor('grey')
        collection.set_linewidth(0)

    if climatology:
        clim_cyclic, _ = add_cyclic_point(climatology, coord=data[lon])
        c2 = ax.contour(lon_cyclic, data[lat], clim_cyclic, levels=levels2, colors='m', alpha=0.9, linewidths=.5,
                        transform=ccrs.PlateCarree())
        ax.clabel(c2, colors='k', inline=True, fontsize=5)
    # Set the title
    #ax.set_title(title, fontsize=7)
    ax.text(-0.1, 1.1, sequence, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    ax.text(0.7, 1.1, exp, transform=ax.transAxes, fontsize=7, va='top', ha='left')

    return cf

def plot_trend_and_significance_ssp(axes, data, trend, p_values, title, sequence, exp, levels, levels2,
                                    ccmap, climatology=None):
    ax = fig.add_axes(axes, projection=ccrs.Robinson())
    ax.set_global()
    ax.set_aspect(1.25)
    # 为数据添加周期性点
    trend_cyclic, lon_cyclic = add_cyclic_point(trend, coord=data['lon'])
    # 确保一致性掩码也添加周期性点
    consistency_mask_cyclic, _ = add_cyclic_point(p_values, coord=data['lon'])
    # Plot the trend
    cf = ax.contourf(lon_cyclic, data['lat'], trend_cyclic, levels=levels, cmap=ccmap, extend='both',
                     transform=ccrs.PlateCarree())
    ax.coastlines(lw=0.5)
    # Add gridlines and labels
    gl = ax.gridlines(draw_labels=True, color='black', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.bottom_labels = True
    gl.left_labels = True
    gl.right_labels = False
    gl.xlabel_style = {'size': 7}
    gl.ylabel_style = {'size': 7}
    # Plot the significance
    significance = np.ma.masked_where(p_values >= 0.05, p_values)
    cc = ax.contourf(data['lon'], data['lat'], significance, hatches=['...'], colors='none',
                     transform=ccrs.PlateCarree())
    for j, collection in enumerate(cc.collections):  ############更改打点的颜色
        collection.set_edgecolor('grey')
    for collection in cc.collections:
        collection.set_linewidth(0)

    if climatology:
        clim_cyclic, _ = add_cyclic_point(climatology, coord=data['lon'])
        c2 = ax.contour(lon_cyclic, data['lat'], clim_cyclic, levels=levels2, colors='m', alpha=1, linewidths=.5,
                        transform=ccrs.PlateCarree())
        ax.clabel(c2, colors='k', inline=True, fontsize=5)
    # Set the title
    #ax.set_title(title, fontsize=7)
    ax.text(-0.1, 1.1, sequence, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    ax.text(.7, 1.1, exp, transform=ax.transAxes, fontsize=7, va='top', ha='left')
    return cf


def plot_zonmean_diff(ax, trend_zonmean, linesty, lat, title, sequence, xlim,std_diff=None,linecolor=None):
    # 计算100hPa和200hPa的纬向风差值
    latitudes = lat.values
    if linecolor:
        linec = linecolor
    else:
        linec = 'k'
    # 绘制集合平均
    trend_zonmean_line = ax.plot(trend_zonmean, latitudes, linec,linestyle=linesty, linewidth=1)
    if std_diff is not None:
    # 填充模型间一倍标准差
        ax.fill_betweenx(latitudes, trend_zonmean - std_diff, trend_zonmean + std_diff, color=linec, alpha=0.2)
    ax.fill_between(xlim, -20, -40, color='dodgerblue', alpha=0.02)
    ax.fill_between(xlim, 20, 40, color='dodgerblue', alpha=0.02)
    # 绘制竖直0刻度线
    ax.axvline(0, color='grey', linewidth=.5)
    # 设置横纵坐标范围
    ax.set_xlim(xlim)
    ax.set_ylim([-90, 90])
    #ax.xaxis.set_major_locator(MultipleLocator(2))
    #ax.yaxis.set_major_locator(MultipleLocator(0.4))
    # 设置纬度标签
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_yticklabels(['90°S', '60°S', '30°S', '0°', '30°N', '60°N', '90°N'])
    ax.tick_params(axis='both', which='major', direction='in',labelsize=7)
    # 设置标题
    #ax.set_title(title, fontsize=7)
    # 设置坐标轴标签
    ax.text(-.28, 1.1, sequence, transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')
    return trend_zonmean_line

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

    ax.set_title(title, loc='right', fontsize=12)
    ax.set_ylabel('hPa', fontsize=12)
    ax.text(-0.2, 1.1, sequence, transform=ax.transAxes, fontsize=22, fontweight='bold', va='top', ha='left')
    return cc
