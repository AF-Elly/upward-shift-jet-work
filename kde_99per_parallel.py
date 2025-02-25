import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.stats import gaussian_kde
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.colors import BoundaryNorm
from cartopy.util import add_cyclic_point
import dask.array as da
from joblib import Parallel, delayed

# 2. 筛选出纬度范围在 20-60 度之间的数据
def filter_def_by_latitude(def_plev9, lat, lat_min=20, lat_max=60):
    # 使用 Dask 来处理大数据
    lat_indices = np.where((lat >= lat_min) & (lat <= lat_max))[0]
    def_filtered = def_plev9[:, lat_indices, :]

    # 筛选出对应的纬度数组
    lat_filtered = lat[lat_indices]

    return def_filtered, lat_filtered

# 计算高斯核密度估计的 CDF (3.compute_def_threshold_filtered的子函数parallel_cdf_computation 的子函数)
def compute_kde_cdf_chunk(chunk, values):
    kde = gaussian_kde(chunk)
    return np.cumsum(kde(values)) / np.sum(kde(values))

# 将整个数据集分割成多个块进行并行计算，提高计算kde_cdf的效率 (3.compute_def_threshold_filtered的子函数)
def parallel_cdf_computation(def_flat_noinf, values, num_chunks=64):
    # 将数据分割成 num_chunks 块
    chunk_size = len(def_flat_noinf) // num_chunks
    chunks = [def_flat_noinf[i:i + chunk_size] for i in range(0, len(def_flat_noinf), chunk_size)]
    print('cdf_chunks set up!')
    # 使用 joblib 并行计算每个块的 CDF （n_jobs=并行数，调用核心数）
    cdf_chunks = Parallel(n_jobs=num_chunks)(delayed(compute_kde_cdf_chunk)(chunk, values) for chunk in chunks)
    print('cdf_chunks done!')
    # 合并所有的 CDF 结果
    cdf = np.sum(cdf_chunks, axis=0)
    #cdf /= np.sum(cdf_chunks[0])  # 归一化

    # 对合并后的 CDF 进行归一化，确保最大值为 1
    cdf /= np.max(cdf)  # 将 CDF 归一化到 [0, 1]
    return cdf

# 3. 计算99%分位点对应的值
def compute_def_threshold_filtered(def_plev9, lat, lat_min=20, lat_max=60, num_chunks=64):
    # 筛选出纬度范围在 20-60 度之间的数据
    def_filtered, lat_filtered = filter_def_by_latitude(def_plev9, lat, lat_min, lat_max)

    # 将数据展平成一维，计算概率密度分布
    def_flat = def_filtered.flatten()

    # 去除无穷值
    def_flat_noinf = np.delete(def_flat, np.where(np.isinf(def_flat)))
    def_flat_noinf = np.delete(def_flat_noinf, np.where(np.isnan(def_flat_noinf)))

    # 计算99%分位点
    values = np.linspace(min(def_flat_noinf), max(def_flat_noinf), 1000)

    # 并行计算 CDF
    cdf = parallel_cdf_computation(def_flat_noinf, values, num_chunks=num_chunks)
   # print(cdf)
    #print(type(cdf))
    # 找到99%分位点对应的值
    threshold_index = np.where(cdf >= 0.99)[0][0]
    threshold_value = values[threshold_index]
    #print(threshold_value)
    #print(type(threshold_value))
    return threshold_value, cdf

# 4. 计算湍流发生的概率（TI1超过给定阈值的概率）
def compute_turbulence_probability(def_plev9, threshold):
    # 对每个格点，计算在所有时间中超过阈值的次数
    turbulence_occurrences = np.sum(def_plev9 > threshold, axis=0)

    # 计算湍流发生的概率（以百分比表示）
    probability = (turbulence_occurrences / def_plev9.shape[0]) * 100

    return probability


# 示例并行计算函数调用
def parallel_turbulence_analysis(index, lat, num_chunks=64):

    turbulence_probability_list = []
    for lev in range(index.shape[1]):
        print(lev)
        # 计算第99%分位点的 DEF 阈值
        threshold, cdf = compute_def_threshold_filtered(index[:, lev], lat, num_chunks=num_chunks)
        print('threshold done!')
        # 计算湍流发生的概率
        probability = compute_turbulence_probability(index[:, lev], threshold)
        turbulence_probability_list.append(probability)

    turbulence_probability_array = np.stack(turbulence_probability_list)
    return turbulence_probability_array


# "ACCESS-CM2" VWS DEF最高两层都是空值？
file_path1 = f'/home/share-from-2/dongyl/cmip_day/interpolated_remapbiled_289x145/N_hemisphere/ua_historical_FGOALS-g3_day_interpolated_289x145_N.nc'
# file_path2 = f'/home/dongyl/Work2024/Tl/deal_Tl_data/va_interpolated/future/N_hemsphere/va_ssp585_{model}_day_interpolated_289x145_N.nc'
# file_path3 = f'/home/dongyl/Work2024/Tl/deal_Tl_data/zg_interpolated/future/N_hemsphere/zg_ssp585_{model}_day_interpolated_289x145_N.nc'
lat = xr.open_dataset(file_path1).lat.values
lat_indices = np.where((lat >= 10) & (lat <= 80))[0]
lat = xr.open_dataset(file_path1).lat[lat_indices].values
models = ["BCC-CSM2-MR", "CanESM5", "CESM2-WACCM", "FGOALS-g3",
          "GFDL-CM4", "IITM-ESM","INM-CM4-8",'INM-CM5-0',"IPSL-CM6A-LR",
          "KACE-1-0-G","MIROC6", "MPI-ESM1-2-HR", "MPI-ESM1-2-LR",
          "MRI-ESM2-0","NorESM2-LM","NorESM2-MM","TaiESM1"]
for model in models:
    VWS = np.load(f'/home/dongyl/Work2024/Tl/obs_npy/VWS_historical_{model}_1980-2010_N.npy')
    #DEF = np.load(f'/home/dongyl/Work2024/Tl/obs_npy/DEF_historical_{model}_1980-2010_N.npy')
    index =  np.log(VWS)
    print('ok')
    # 并行计算湍流指数和概率
    turbulence_probability_array = parallel_turbulence_analysis(index, lat)
    # 保存概率结果
    np.save(f'/home/dongyl/Work2024/Tl/obs_npy/logVWS_historical_{model}_1980-2010_N_frequency_500-10hpa.npy',
            turbulence_probability_array)
