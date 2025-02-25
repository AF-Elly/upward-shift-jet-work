import xarray as xr
import numpy as np
import metpy.calc as mpcalc
from metpy.units import units
import dask.array as da

# 加载数据
model = "FGOALS-g3"
models = ["BCC-CSM2-MR", "CanESM5", "CESM2-WACCM", "FGOALS-g3", "GFDL-CM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR",
          "MPI-ESM1-2-LR","KACE-1-0-G","IITM-ESM",
          "NorESM2-MM", "MIROC6",
          "NorESM2-LM", "INM-CM4-8", "INM-CM5-0", "TaiESM1",
          "MRI-ESM2-0"]  # "KACE-1-0-G" DEF似乎没有,"IITM-ESM"

ds_u = xr.open_dataset('/home/share-from-2/dongyl/cmip_day/interpolated_remapbiled_289x145/N_hemisphere/ua_ssp585_BCC-CSM2-MR_day_interpolated_289x145_N.nc')
# 提取数据
lat = ds_u.lat.values
lat_indices = np.where((lat >= 10) & (lat <= 80))[0]

# 定义单位
lat_units = units.degrees_north
lon_units = units.degrees_east

# 使用 MetPy 计算梯度
def calculate_DEF(u,v):
    # 计算 u 和 v 在纬度和经度方向的梯度
    #du_dx, du_dy = mpcalc.gradient(u)  # u 在 lat 和 lon 方向的梯度
    #dv_dx, dv_dy = mpcalc.gradient(v)  # v 在 lat 和 lon 方向的梯度
    #u_np = u.values
    #v_np = v.values
    #du_dx, du_dy = mpcalc.gradient(u_np * lat_units, axes=(2, 3))
    #dv_dx, dv_dy = mpcalc.gradient(v_np * lon_units, axes=(2, 3))
    du_dx, du_dy = mpcalc.gradient(u * lat_units, axes=(2, 3))
    dv_dx, dv_dy = mpcalc.gradient(v * lon_units, axes=(2, 3))
    du_dx = du_dx.values
    dv_dx = dv_dx.values
    du_dy = du_dy.values
    dv_dy = dv_dy.values
    # 计算 DEF
    DEF = np.sqrt((dv_dx + du_dy) ** 2 + (du_dx - dv_dy) ** 2)
    return DEF

for model in models:
    print(f'{model} begin')
    file_path1 = f'/home/share-from-2/dongyl/cmip_day/interpolated_remapbiled_289x145/N_hemisphere/ua_ssp585_{model}_day_interpolated_289x145_N.nc'
    file_path2 = f'/home/share-from-2/dongyl/cmip_day/interpolated_remapbiled_289x145/N_hemisphere/va_ssp585_{model}_day_interpolated_289x145_N.nc'

    # 使用 Dask 加载数据集
    ds_u = xr.open_dataset(file_path1, chunks={'time': 100, 'lat': 20, 'lon': 20}, engine="netcdf4")
    ds_v = xr.open_dataset(file_path2, chunks={'time': 100, 'lat': 20, 'lon': 20}, engine="netcdf4")

    # 提取数据
    #lat = ds_u.lat.values
    #lat_indices = np.where((lat >= 10) & (lat <= 80))[0]

    u = ds_u.ua_interp[:, 4:, lat_indices]
    v = ds_v.va_interp[:, 4:, lat_indices]

    lat = ds_u.lat[lat_indices].values
    lon = ds_u.lon.values

    # 将 u 和 v 转换为 xarray.DataArray 并指定坐标
    u_xr = xr.DataArray(u, coords=[ds_u.time, ds_u.plev[4:], ds_u.lat[lat_indices], ds_u.lon],
                        dims=["time", "plev", "lat", "lon"], name="u")
    v_xr = xr.DataArray(v, coords=[ds_v.time, ds_v.plev[4:], ds_v.lat[lat_indices], ds_v.lon],
                        dims=["time", "plev", "lat", "lon"], name="v")



    # 计算 DEF
    DEF = calculate_DEF(u_xr,v_xr)

    # 执行并行计算
    #DEF.compute()

    # 将结果保存到 .npy 文件
    np.save(f'/home/dongyl/Work2024/Tl/obs_npy/DEF_ssp585_{model}_2060-2090_N_new.npy', DEF)
    print(f'{model} DEF saved!')