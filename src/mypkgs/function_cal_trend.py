import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
import numpy as np
import xarray as xr
from scipy.stats import linregress


def calculate_trend_3D_ndarray(da):
    """
    计算三维数据数组中每个模式的每个纬度上的时间序列趋势。

    参数:
    da (numpy.ndarray): 三维数据数组，维度为模式，时间，纬度。

    返回:
    tuple: 包含趋势和p值的两个数组。
    """
    # 提取维度信息
    time_dim, lat_dim, lon_dim = da.shape

    # 初始化趋势和p值数组
    trend = np.zeros((lat_dim, lon_dim))
    p_value = np.zeros((lat_dim, lon_dim))

    # 遍历每个模式和每个纬度
    for lat_index in range(lat_dim):
        for lon_index in range(lon_dim):
            # 提取时间序列
            y = da[:, lat_index, lon_index]

            # 使用linregress计算趋势和p值
            slope, intercept, r_value, p_val, std_err = stats.linregress(np.arange(time_dim), y)

            # 保存结果
            trend[lat_index, lon_index] = slope
            p_value[lat_index, lon_index] = p_val

    return trend, p_value
def calculate_trend_4D_ndarray(da):
    """
    计算四维数据数组中每个模式的每个纬度上的时间序列趋势。

    参数:
    da (numpy.ndarray): 四维数据数组，维度为时间，气压层，经度，纬度。

    返回:
    tuple: 包含趋势和p值的两个数组。
    """
    # 提取维度信息
    time_dim, plev_dim, lat_dim, lon_dim = da.shape

    # 初始化趋势和p值数组
    trend = np.zeros((plev_dim, lat_dim, lon_dim))
    p_value = np.zeros((plev_dim, lat_dim, lon_dim))

    # 遍历每个模式和每个纬度
    for plev_index in range(plev_dim):
        for lat_index in range(lat_dim):
            for lon_index in range(lon_dim):
                # 提取时间序列
                y = da[:, plev_index, lat_index, lon_index]

                # 使用linregress计算趋势和p值
                slope, intercept, r_value, p_val, std_err = stats.linregress(np.arange(time_dim), y)

                # 保存结果
                trend[plev_index, lat_index, lon_index] = slope
                p_value[plev_index, lat_index, lon_index] = p_val

    return trend, p_value

def get_slope_p_2D(y,x):
    year = x
    A = np.vstack([year, np.ones(len(year))]).T
    y_T= y.data.reshape(y.shape[0], 1)
    slope = np.linalg.lstsq(A, y_T, rcond=-1)[0][0]
    p_value = f_regression(np.nan_to_num(y_T), year)[1]

    return slope, p_value

def calculate_trend(da):
    a, b = da.dims
    trend = np.zeros(da[a].shape[0])
    p_value = np.zeros(da[a].shape[0])

    for i in range(0, da[a].shape[0]):
        trend[i], intercept, r_value, p_value[i], std_err = \
                stats.linregress(np.arange(da[b].shape[0]), da[i])

    return trend, p_value

def calculate_trend_2D(da):
    time, lat = da.dims
    trend = np.zeros(da[lat].shape[0])
    p_value = np.zeros(da[lat].shape[0])

    for i in range(0, da[lat].shape[0]):
        trend[i], intercept, r_value, p_value[i], std_err = \
                stats.linregress(np.arange(da[time].shape[0]), da[:,i])

    return trend, p_value

def calculate_trend_2D_zonmean(da):
    """
    计算三维数据数组中每个模式的每个纬度上的时间序列趋势。
    参数:
    da (numpy.ndarray): 三维数据数组，维度为模式，时间，纬度。

    返回:
    tuple: 包含趋势和p值的两个数组。
    """
    # 提取维度信息
    time_dim, lat_dim = da.shape

    # 初始化趋势和p值数组
    trend = np.zeros(lat_dim)
    p_value = np.zeros(lat_dim)

    # 遍历每个模式和每个纬度
    for lat_index in range(lat_dim):
            # 提取时间序列
        y = da[:, lat_index]
        # 使用linregress计算趋势和p值
        slope, intercept, r_value, p_val, std_err = stats.linregress(np.arange(time_dim), y)
        # 保存结果
        trend[lat_index] = slope
        p_value[lat_index] = p_val
    return trend, p_value

def calculate_trend_2D_sc(da):
    """
    计算三维数据数组中每个模式的每个纬度上的时间序列趋势。
    参数:
    da (numpy.ndarray): 三维数据数组，维度为模式，时间，纬度。

    返回:
    tuple: 包含趋势和p值的两个数组。
    """
    # 提取维度信息
    month_dim, time_dim= da.shape

    # 初始化趋势和p值数组
    trend = np.zeros(month_dim)
    p_value = np.zeros(month_dim)

    # 遍历每个模式和每个纬度
    for month_index in range(month_dim):
            # 提取时间序列
        y = da[month_index, :]
        # 使用linregress计算趋势和p值
        slope, intercept, r_value, p_val, std_err = stats.linregress(np.arange(time_dim), y)
        # 保存结果
        trend[month_index] = slope
        p_value[month_index] = p_val
    return trend, p_value

def calculate_trend_3D_zonmean(da):
    """
    计算三维数据数组中每个模式的每个纬度上的时间序列趋势。

    参数:
    da (numpy.ndarray): 三维数据数组，维度为模式，时间，纬度。

    返回:
    tuple: 包含趋势和p值的两个数组。
    """
    # 提取维度信息
    model_dim, time_dim, lat_dim = da.shape

    # 初始化趋势和p值数组
    trend = np.zeros((model_dim, lat_dim))
    p_value = np.zeros((model_dim, lat_dim))

    # 遍历每个模式和每个纬度
    for model_index in range(model_dim):
        for lat_index in range(lat_dim):
            # 提取时间序列
            y = da[model_index, :, lat_index]

            # 使用linregress计算趋势和p值
            slope, intercept, r_value, p_val, std_err = stats.linregress(np.arange(time_dim), y)

            # 保存结果
            trend[model_index, lat_index] = slope
            p_value[model_index, lat_index] = p_val

    return trend, p_value

def calculate_trend_3D_sc(da):
    """
    计算三维数据数组中每个模式的每个纬度上的时间序列趋势。

    参数:
    da (numpy.ndarray): 三维数据数组，维度为模式，时间，纬度。

    返回:
    tuple: 包含趋势和p值的两个数组。
    """
    # 提取维度信息
    model_dim, month_dim, time_dim = da.shape

    # 初始化趋势和p值数组
    trend = np.zeros((model_dim, month_dim))
    p_value = np.zeros((model_dim, month_dim))

    # 遍历每个模式和每个纬度
    for model_index in range(model_dim):
        for month_index in range(month_dim):
            # 提取时间序列
            y = da[model_index, month_index, :]

            # 使用linregress计算趋势和p值
            slope, intercept, r_value, p_val, std_err = stats.linregress(np.arange(time_dim), y)

            # 保存结果
            trend[model_index, month_index] = slope
            p_value[model_index, month_index] = p_val

    return trend, p_value


def calculate_trend_4D_zonmean(da):
    """
    计算三维数据数组中每个模式的每个纬度上的时间序列趋势。

    参数:
    da (numpy.ndarray): 三维数据数组，维度为模式，时间，纬度。

    返回:
    tuple: 包含趋势和p值的两个数组。
    """
    # 提取维度信息
    model_dim, month_dim, time_dim, lat_dim = da.shape

    # 初始化趋势和p值数组
    trend = np.zeros((model_dim, month_dim,lat_dim))
    p_value = np.zeros((model_dim, month_dim,lat_dim))

    # 遍历每个模式和每个纬度
    for model_index in range(model_dim):
        for month_index in range(month_dim):
            for lat_index in range(lat_dim):
            # 提取时间序列
                y = da[model_index, month_index, :, lat_index]

            # 使用linregress计算趋势和p值
                slope, intercept, r_value, p_val, std_err = stats.linregress(np.arange(time_dim), y)

            # 保存结果
                trend[model_index, month_index, lat_index] = slope
                p_value[model_index, month_index, lat_index] = p_val

    return trend, p_value

# 示例用法
# 假设你有一个名为da的三维数组
# trend, p_value = calculate_trend(da)
# 现在，trend和p_value分别包含了趋势和p值信息
def get_slope_p_3D(da):
    time, lat, lon = da.dims
    year = np.arange(1979, 1979+len(da))
    A = np.vstack([year, np.ones(len(year))]).T
    ds_2d = da.data.reshape(da.data.shape[0], da.data.shape[1] * da.data.shape[2])
    slope = np.linalg.lstsq(A, ds_2d, rcond=-1)[0][0].reshape(len(da[lat]), len(da[lon]))
    p_value = f_regression(np.nan_to_num(ds_2d), year)[1].reshape(len(da[lat]), len(da[lon]))

    return slope, p_value

def get_slope_p_4D(da):
    """
    计算四维数据数组中每个模式的每个经纬度格点上的时间序列的趋势和显著性。

    参数:
    da (xarray.DataArray): 四维数据数组，维度为模式，时间，纬度，经度。

    返回:
    xarray.Dataset: 包含趋势（斜率）和p值的数据集。
    """
    # 提取维度名称
    modes, time, lat, lon = da.dims

    # 初始化输出数组
    slope = xr.DataArray(np.zeros((len(da[modes]), len(da[lat]), len(da[lon]))),
                         dims=[modes, lat, lon],
                         coords={modes: da[modes], lat: da[lat], lon: da[lon]})
    p_value = xr.DataArray(np.zeros((len(da[modes]), len(da[lat]), len(da[lon]))),
                           dims=[modes, lat, lon],
                           coords={modes: da[modes], lat: da[lat], lon: da[lon]})

    # 遍历每个模式和每个经纬度点
    for i, mode in enumerate(da[modes].values):
        for j, latitude in enumerate(da[lat].values):
            for k, longitude in enumerate(da[lon].values):
                # 提取时间序列
                y = da.sel({modes: mode, lat: latitude, lon: longitude}).values

                # 计算时间
                x = np.arange(len(y))

                # 忽略全为NaN的情况
                if np.isnan(y).all():
                    slope[i, j, k] = np.nan
                    p_value[i, j, k] = np.nan
                else:
                    # 计算斜率和p值
                    res = linregress(x, y)
                    slope[i, j, k] = res.slope
                    p_value[i, j, k] = res.pvalue

    return xr.Dataset({'slope': slope, 'p_value': p_value})

def get_slope_p_4D_tplatlon(da):
    """
    计算四维数据数组中每个模式的每个经纬度格点上的时间序列的趋势和显著性。

    参数:
    da (xarray.DataArray): 四维数据数组，维度为模式，时间，纬度，经度。

    返回:
    xarray.Dataset: 包含趋势（斜率）和p值的数据集。
    """
    # 提取维度名称
    time, plev, lat, lon = da.dims

    # 初始化输出数组
    slope = xr.DataArray(np.zeros((len(da[plev]), len(da[lat]), len(da[lon]))),
                         dims=[plev, lat, lon],
                         coords={plev: da[plev], lat: da[lat], lon: da[lon]})
    p_value = xr.DataArray(np.zeros((len(da[plev]), len(da[lat]), len(da[lon]))),
                           dims=[plev, lat, lon],
                           coords={plev: da[plev], lat: da[lat], lon: da[lon]})

    # 遍历每个模式和每个经纬度点
    for i, p in enumerate(da[plev].values):
        for j, latitude in enumerate(da[lat].values):
            for k, longitude in enumerate(da[lon].values):
                # 提取时间序列
                y = da.sel({plev: p, lat: latitude, lon: longitude}).values

                # 计算时间
                x = np.arange(len(y))

                # 忽略全为NaN的情况
                if np.isnan(y).all():
                    slope[i, j, k] = np.nan
                    p_value[i, j, k] = np.nan
                else:
                    # 计算斜率和p值
                    res = linregress(x, y)
                    slope[i, j, k] = res.slope
                    p_value[i, j, k] = res.pvalue

    return xr.Dataset({'slope': slope, 'p_value': p_value})