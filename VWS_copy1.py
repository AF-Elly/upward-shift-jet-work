import numpy as np
import xarray as xr
from joblib import Parallel, delayed
'''
# 计算垂直风切变 (VWS) 遍历循环计算
def compute_vertical_wind_shear(u, v, zg):
    # 初始化与 u, v 形状相同的 du/dz 和 dv/dz 数组
    du_dz = np.zeros_like(u)
    dv_dz = np.zeros_like(v)

    # 对每个 (time, lat, lon) 网格点，沿 plev 维度计算 du/dz 和 dv/dz
    for t in range(u.shape[0]):  # 遍历时间维度
        for i in range(u.shape[2]):  # 遍历纬度维度
            for j in range(u.shape[3]):  # 遍历经度维度
                # 取出当前时间、纬度、经度点的 u, v, zg 值
                u_plev = u[t, :, i, j]  # 取出该网格点的所有气压层上的 u 值
                v_plev = v[t, :, i, j]  # 取出该网格点的所有气压层上的 v 值
                zg_plev = zg[t, :, i, j]  # 取出该网格点的所有气压层上的 zg 值

                # 对气压层维度 (plev) 的 zg 进行梯度计算
                du_dz[t, :, i, j] = np.gradient(u_plev, zg_plev)
                dv_dz[t, :, i, j] = np.gradient(v_plev, zg_plev)

    # 计算垂直风切变 (VWS)
    VWS = np.sqrt(du_dz ** 2 + dv_dz ** 2)

    return VWS
'''
#######使用并行计算的新函数，2024.11.30更新！运行12min左右#########
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# 定义一个计算单个 (t, i, j) 网格点的梯度的函数
def compute_single_grid(t, i, j):
    u_plev = u[t, :, i, j]
    v_plev = v[t, :, i, j]
    zg_plev = zg[t, :, i, j]

    # 计算 du/dz 和 dv/dz
    du_dz_value = np.gradient(u_plev, zg_plev)
    dv_dz_value = np.gradient(v_plev, zg_plev)
    #print(f"Processed time step {t}, latitude {i}, longitude {j}")
    return t, i, j, du_dz_value, dv_dz_value

# 计算垂直风切变 (VWS)
def compute_vertical_wind_shear(u, v, zg):
    # 初始化与 u, v 形状相同的 du/dz 和 dv/dz 数组
    du_dz = np.zeros_like(u)
    dv_dz = np.zeros_like(v)

    # 使用ProcessPoolExecutor并行化计算任务,64个核心非常快！
    with ProcessPoolExecutor(max_workers=4) as executor:
        '''
        batch_size = 1000  # 每次提交的批量大小
        futures = []
        times = u.shape[0]
        lats = u.shape[2]
        lons = u.shape[3]
        for t in range(times):
            print(f"Processed time step {t}")
            for i in range(lats):
                for j in range(lons):
                    futures.append(executor.submit(compute_single_grid, t, i, j))
                    if len(futures) >= batch_size:
                        # 等待当前批次任务完成
                        for future in as_completed(futures):
                            t, i, j, du_dz_value, dv_dz_value = future.result()
                            du_dz[t, :, i, j] = du_dz_value
                            dv_dz[t, :, i, j] = dv_dz_value
                        futures.clear()  # 清空已处理的任务
        '''
        futures = []
        for t in range(u.shape[0]):  # 遍历时间维度
            print(f"computing t={t}")
            for i in range(u.shape[2]):  # 遍历纬度维度
                for j in range(u.shape[3]):  # 遍历经度维度
                    futures.append(executor.submit(compute_single_grid, t, i, j))

    
        # 获取并处理计算结果
        for future in as_completed(futures):
            t, i, j, du_dz_value, dv_dz_value = future.result()
            du_dz[t, :, i, j] = du_dz_value
            dv_dz[t, :, i, j] = dv_dz_value


    VWS = np.sqrt(du_dz ** 2 + dv_dz ** 2)
    return VWS

# 示例数据
#models = ["MPI-ESM1-2-LR", "NorESM2-MM", "MIROC6", "NorESM2-LM"]
models = ["INM-CM4-8"]#MPI-ESM1-2-HR运行中
for model in models:
    file_path1 = f'/home/dongyl/Work2024/Tl/deal_Tl_data/ua_interpolated/ua_historical_{model}_day_interpolated_289x145_N.nc'
    file_path2 = f'/home/dongyl/Work2024/Tl/deal_Tl_data/va_interpolated/va_historical_{model}_day_interpolated_289x145_N.nc'
    file_path3 = f'/home/dongyl/Work2024/Tl/deal_Tl_data/zg_interpolated/zg_historical_{model}_day_interpolated_289x145_N.nc'

    # 读取数据
    ds_u = xr.open_dataset(file_path1)
    ds_v = xr.open_dataset(file_path2)
    ds_zg = xr.open_dataset(file_path3)
    # 分块读取数据，避免一次性读取全部数据
    # ds_u = xr.open_dataset(file_path1, chunks={'time': 100, 'lat': 20, 'lon': 20})  # 设置合理的块大小
    # ds_v = xr.open_dataset(file_path2, chunks={'time': 100, 'lat': 20, 'lon': 20})
    # ds_zg = xr.open_dataset(file_path3, chunks={'time': 100, 'lat': 20, 'lon': 20})
    print('load datas done')
    # 获取纬度和经度信息
    lat = ds_u.lat.values
    lat_indices = np.where((lat >= 10) & (lat <= 80))[0]
    u = ds_u.ua_interp[:, 4:, lat_indices].values
    v = ds_v.va_interp[:, 4:, lat_indices].values
    zg = ds_zg.zg_interp[:, 4:, lat_indices].values
    lat = ds_u.lat[lat_indices].values
    lon = ds_u.lon.values
    print('load datas done2')
    # 调用函数
    VWS = compute_vertical_wind_shear(u, v, zg)
    np.save(f'/home/dongyl/Work2024/Tl/obs_npy/VWS_historical_{model}_1980-2010_N.npy', VWS)




