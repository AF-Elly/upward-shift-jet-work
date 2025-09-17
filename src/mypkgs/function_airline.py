import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import BoundaryNorm
from cartopy.util import add_cyclic_point
#import dyl_function_slope as dyl
import matplotlib.patches as patches
import cmaps
from geopy.distance import great_circle
import math
from cartopy.feature import ShapelyFeature
from cartopy import geodesic
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

def calculate_consistency_2d(model_trends, ensemble_trend, agreement):
    num_models = model_trends.shape[0]
    consistency_mask = np.full(ensemble_trend.shape, False)
    for i in range(ensemble_trend.shape[0]):
        for j in range(ensemble_trend.shape[1]):
            # 计算与集合平均趋势符号一致的模式数量
            num_consistent_models = np.sum(np.sign(model_trends[:, i, j]) == np.sign(ensemble_trend[i, j]))
            # 判断是否超过80%
            if num_consistent_models / num_models >= agreement:
                consistency_mask[i, j] = True
    return consistency_mask

# 绘制湍流的历史趋势
def plot_trend(ax, lat, lon, trend, trend_level, consistency,  left_title, right_title,sequence,sequence_size=18,
               wind_speed_clim_mme=None,bbox_list=None,set_aspect_num=None):
    central_lon = 0
    ax = ax
    ax.set_extent([-180, 180, 0, 80], crs=ccrs.PlateCarree())
    ax.projection = ccrs.PlateCarree(central_longitude=central_lon)
    # 绘制海岸线
    ax.coastlines(linewidth = 0.7)
    if set_aspect_num:
        ax.set_aspect(set_aspect_num)
    # 绘制网格线
    ax.gridlines(draw_labels=False, linestyle='--', color='gray', alpha=0.7, linewidth=.5)

    #levels = np.arange(0.5, 4, 0.25)
    norm = BoundaryNorm(trend_level, ncolors=256, extend='both')
    #lon,lat = np.meshgrid(lon, lat)
    cyc_trend,cyc_lon= add_cyclic_point(trend, coord=lon)

    # 使用 contourf 绘制等值线图，并使用经纬度数据
    contour = ax.contourf(cyc_lon, lat,cyc_trend, transform=ccrs.PlateCarree(), cmap=cmaps.NEO_div_vegetation_a,
                          levels = trend_level, extend='both')

    #绘制气候态风场
    #wind_speed = calculate_wind_speed(u, v, 9)
    if wind_speed_clim_mme:
    # 绘制风速大小的等值线，间隔为10
        wind_levels = np.arange(0, 51, 5)
        cyc_wind_speed, cyc_lon = add_cyclic_point(wind_speed_clim_mme, coord=lon)
        wind_contour = ax.contour(cyc_lon, lat, cyc_wind_speed, levels=wind_levels, colors='k', transform=ccrs.PlateCarree(),linewidths=.7)
        # 为风速等值线添加标签
        ax.clabel(wind_contour, fmt='%d', inline=True, fontsize=8, colors='k')


    # 确保一致性掩码也添加周期性点
    consistency_mask_cyclic, _ = add_cyclic_point(consistency, coord=lon)
    # 在plot_trend_and_significance函数内部，在绘制显著性的代码块之后，添加以下代码来绘制一致性阴影
    consistency = np.ma.masked_where(~consistency_mask_cyclic, cyc_trend)
    cc=ax.contourf(cyc_lon, lat, consistency, hatches=['..'], colors='none', transform=ccrs.PlateCarree())
    '''for j, collection in enumerate(cc.collections):  ############更改打点的颜色
        collection.set_edgecolor('grey')
    for collection in cc.collections:
        collection.set_linewidth(0)'''
    # 检查是否有 `collections` 属性，否则尝试直接访问
    if hasattr(cc, 'collections'):
        collections = cc.collections
    else:
        # 如果 `cc` 本身就是类似集合的对象（旧版本 Matplotlib）
        collections = [cc]  # 包装成列表，使其可迭代
    # 设置 hatch 样式
    for collection in collections:
        collection.set_edgecolor('grey')
        collection.set_linewidth(0)

    # 设置规范的经纬度格式
    ax.set_xticks(np.arange(-180, 181, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(0, 80, 20), crs=ccrs.PlateCarree())
    ax.set_ylim(10, 81)

    # 设置坐标轴格式为经纬度
    lon_formatter = LongitudeFormatter()
    lat_formatter = LatitudeFormatter()

    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)

    # 隐藏上侧和右侧的坐标轴刻度
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 只保留左侧和下侧的 tick
    ax.tick_params(top=False, right=False)

    # 设置标题
    plt.title(left_title, loc='left', fontsize=10, pad=4)
    plt.title(right_title, loc='right', fontsize=10, pad=4)
    ax.text(-0.05, 1.2, sequence, transform=ax.transAxes, fontsize=sequence_size, fontweight='bold', va='top', ha='left')
    #plt.savefig(ppath, format='png',dpi=500)
    # **绘制多个方框（如果提供了 bbox_list）**
    if bbox_list:
        for bbox in bbox_list:
            min_lon, max_lon, min_lat, max_lat = bbox
            width = max_lon - min_lon
            height = max_lat - min_lat
            rect = patches.Rectangle((min_lon, min_lat), width, height, linewidth=1,
                                     edgecolor='blue', facecolor='none', linestyle='-',
                                     transform=ccrs.PlateCarree())
            ax.add_patch(rect)
    return ax,contour

def plot_bbox(ax, bbox_list=None):
    # **绘制多个方框（如果提供了 bbox_list）**
    if bbox_list:
        for bbox in bbox_list:
            min_lon, max_lon, min_lat, max_lat = bbox
            width = max_lon - min_lon
            height = max_lat - min_lat
            rect = patches.Rectangle((min_lon, min_lat), width, height, linewidth=1,
                                     edgecolor='blue', facecolor='none', linestyle='-',
                                     transform=ccrs.PlateCarree(),zorder=10000 )# 设为高数值确保在最上层
            ax.add_patch(rect)
    return ax

def calculate_great_circle_path(start_lon, start_lat, end_lon, end_lat, num_points=100):
    """
    计算两点之间的大圆路径经纬度。

    参数：
    start_lon, start_lat (float): 起始点的经纬度
    end_lon, end_lat (float): 终点的经纬度
    num_points (int): 大圆路径上点的数量（默认100点）

    返回：
    lons, lats (list): 大圆路径上的经纬度点
    """
    # 将角度转换为弧度
    start_lat, start_lon = np.radians(start_lat), np.radians(start_lon)
    end_lat, end_lon = np.radians(end_lat), np.radians(end_lon)

    # 计算大圆航线的参数
    delta_lon = end_lon - start_lon
    delta_sigma = np.arccos(np.sin(start_lat) * np.sin(end_lat) +
                            np.cos(start_lat) * np.cos(end_lat) * np.cos(delta_lon))

    # 计算路径上每个点的经纬度
    lons = []
    lats = []
    for i in range(num_points + 1):
        sigma = delta_sigma * i / num_points  # 计算大圆路径上的点
        A = np.sin((1 - i / num_points) * delta_sigma) / np.sin(delta_sigma)
        B = np.sin(i / num_points * delta_sigma) / np.sin(delta_sigma)

        # 利用球面插值公式计算每个点的经纬度
        x = A * np.cos(start_lat) * np.cos(start_lon) + B * np.cos(end_lat) * np.cos(end_lon)
        y = A * np.cos(start_lat) * np.sin(start_lon) + B * np.cos(end_lat) * np.sin(end_lon)
        z = A * np.sin(start_lat) + B * np.sin(end_lat)

        lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
        lon = np.arctan2(y, x)

        lons.append(np.degrees(lon))  # 转换为度
        lats.append(np.degrees(lat))  # 转换为度

    # 转换为 numpy 数组，以便于处理
    #lons = np.array(lons)
    #lats = np.array(lats)
    # 处理跨越180度经线的情况
    #lons, lats = add_cyclic_point(lons, lats)

    return lons, lats


def plot_great_circle(ax, start_lat, start_lon, end_lat, end_lon, start_name=None, end_name=None):
    """
    绘制从起点到终点的大圆航线。

    参数：
    ax (cartopy.axes.Axes): 地图绘制的轴
    start_lat, start_lon (float): 起始点的纬度和经度
    end_lat, end_lon (float): 终点的纬度和经度
    start_name (str): 起始点名称，用于标注（可选）
    end_name (str): 终点名称，用于标注（可选）
    """
    # 计算大圆路径
    lons, lats = calculate_great_circle_path(start_lon, start_lat, end_lon, end_lat)
    #print(lons)
    #print(lats)
    # 绘制大圆航线
    ax.plot(lons, lats, color='#FF7F0E', linewidth=.2, transform=ccrs.Geodetic(),
            alpha=0.05)

    # 2. 检查是否在南半球（纬度为负）
    if start_lat >= 10:
        # 单独绘制起点（红色实心圆点）
        ax.plot(
        start_lon, start_lat,
        marker='o', markersize=.2, color='red',
        transform=ccrs.Geodetic(), alpha=0.15
        )
    if end_lat >= 10:
        # 可选：单独绘制终点（蓝色实心圆点）
        ax.plot(
        end_lon, end_lat,
        marker='o', markersize=.2, color='r',
        transform=ccrs.Geodetic(), alpha=0.15
        )
    '''
    # 为起始点和终点添加名称
    if  end_name == "ICN" or end_name == "CAI":
        ax.text(end_lon - 5, end_lat -5, end_name, transform=ccrs.PlateCarree(), fontsize=6, color='red',alpha=0.8)
    elif  end_name == "LAX":
        ax.text(end_lon - 3, end_lat +1, end_name, transform=ccrs.PlateCarree(), fontsize=6, color='red',alpha=0.8)
    elif  end_name == "MUC":
        ax.text(end_lon +3, end_lat, end_name, transform=ccrs.PlateCarree(), fontsize=6, color='red',alpha=0.8)
    elif  end_name == "PUJ":
        ax.text(end_lon -10, end_lat+2, end_name, transform=ccrs.PlateCarree(), fontsize=6, color='red',alpha=0.8)
    elif  end_name == "BGI":
        ax.text(end_lon +2, end_lat-3, end_name, transform=ccrs.PlateCarree(), fontsize=6, color='red',alpha=0.8)

    else:
        ax.text(end_lon - 5, end_lat + 2, end_name, transform=ccrs.PlateCarree(), fontsize=6, color='red',alpha=0.8)

    #if start_name:
    ax.text(start_lon -5, start_lat + 2, start_name, transform=ccrs.PlateCarree(), fontsize=6, color='red',alpha=0.8)
    #if end_name:
    '''
    return ax

def plot_routes_on_map(ax, airports, routes):
    """
    在地图上绘制多个航线（大圆航线）。

    参数：
    ax (cartopy.axes.Axes): 地图绘制的轴
    airports (dict): 机场字典，键为机场代码，值为经纬度元组
    routes (list): 航线列表，每条航线是一个包含两个机场代码的元组
    """
    for route in routes:
        try:
            start_code, end_code = route
            # 1. 检查机场是否存在
            start_lat, start_lon = airports[start_code]
            end_lat, end_lon = airports[end_code]

            # 2. 检查是否在南半球（纬度为负）
            if start_lat < 0 or end_lat < 0:
                print(f"跳过南半球航线：{route}（起点纬度: {start_lat}, 终点纬度: {end_lat}）")
                continue

            # 绘制大圆航线
            ax = plot_great_circle(ax, start_lat, start_lon, end_lat, end_lon, start_name=start_code, end_name=end_code)
        except KeyError:
            print(f"跳过无效航线：{route}（机场代码不存在）")
            continue  # 跳过当前航线

    return ax