import os, time, dask
import numpy as np
import xarray as xr

DEF_DIR = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_def"
VWS_DIR = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_vws"
OUT_DIR = "/home/dongyl/Work2_2025/ERA5_daily"
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = list(range(1979, 2016+1))
Q = 0.95
LAT_MIN, LAT_MAX = 20, 60

def open_ti1_year(year: int) -> xr.DataArray:
    def trim_to_365(da: xr.DataArray) -> xr.DataArray:
        # 保证“最后一天”真的是时间序列的最后一个
        da = da.sortby("time")
        if da.sizes.get("time", 0) > 365:
            da = da.isel(time=slice(0, 365))
        return da
    time1 = time.time()
    def_da = xr.open_dataset(f"{DEF_DIR}/era5_def_{year}-50-300hpa.nc")["def"]#, chunks=CHUNKS
    vws_da = xr.open_dataset(f"{VWS_DIR}/era5_vws_{year}_50-300hpa.nc")["vws"]
    time2 = time.time()
    time2_1 = time2-time1
    print(f"load {year} DEF and VWS, use {time2_1}")

    # 各自多于 365 天就去掉最后一天
    def_da = trim_to_365(def_da)
    vws_da = trim_to_365(vws_da)
    # 现在严格对齐（应该都 365 天了）
    def_da, vws_da = xr.align(def_da, vws_da, join="exact")
    ti = (def_da * vws_da).astype("float32").transpose("time", "plev", "lat", "lon")
    time3 = time.time()
    time3_2 = time3-time2
    print(f"calculated {year} TI1, use {time3_2}")

    # --- 关键：把每年的 time 统一成相同的相对索引（0..364） ---
    # 可选：把原来的绝对时间保留下来当辅助坐标（不会参与对齐）
    abs_time = ti.time.values
    ti = ti.assign_coords(time=("time", np.arange(365, dtype=np.int16)))
    ti = ti.assign_coords(time_abs=("time", abs_time))  # 仅做参考，不影响 concat
    return ti

thr_plev = xr.open_dataset("/home/dongyl/Work2_2025/ERA5_daily/era5_TI1_threshold_95p_plev.nc")["threshold"]
for year in YEARS:
    TI1 = open_ti1_year(year)
    ti_sel = TI1.where((TI1.lat >= LAT_MIN) & (TI1.lat <= LAT_MAX), drop=True)
    #thr_plev = [3.23817239689106e-08, 4.91091292076362e-08,
    #    6.93908859261683e-08, 1.05593819910155e-07, 1.43321173595723e-07,
    #    2.04577460749533e-07]
    exceed = TI1 > thr_plev
    prob_year = (exceed.mean(dim="time", skipna=True) * 100.0).astype("float32")
    prob_year.name = "frequency"
    print('写入中。。。')
    prob_year.to_netcdf(f"{OUT_DIR}/era5_TI1_exceed_pct_{year}_{int(Q * 100)}p.nc")
    print(f'已经写入{year}频率')
