import os, time, dask
import numpy as np
import xarray as xr
import json
#此代码有懒加载，因此写入文件时才真正计算，会造成写入时间很长
#从运行到写入threhold需要1h
#写入99p需要

DEF_DIR = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_def"
VWS_DIR = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_vws"
OUT_DIR = "/home/dongyl/Work2_2025/ERA5_daily"
os.makedirs(OUT_DIR, exist_ok=True)

YEARS = list(range(1979, 2024+1))
Q = 0.95
LAT_MIN, LAT_MAX = 20, 60
CHUNKS = {"time": 30, "lat": 180, "lon": 360}  # 适配你的存储块

def open_ti1_year(year: int) -> xr.DataArray:
    def trim_to_365(da: xr.DataArray) -> xr.DataArray:
        # 保证“最后一天”真的是时间序列的最后一个
        da = da.sortby("time")
        if da.sizes.get("time", 0) > 365:
            da = da.isel(time=slice(0, 365))
        return da

    def_da = xr.open_dataset(f"{DEF_DIR}/era5_def_{year}-50-300hpa.nc", chunks=CHUNKS)["def"]#, chunks=CHUNKS
    vws_da = xr.open_dataset(f"{VWS_DIR}/era5_vws_{year}_50-300hpa.nc", chunks=CHUNKS)["vws"]

    # 各自多于 365 天就去掉最后一天
    def_da = trim_to_365(def_da)
    vws_da = trim_to_365(vws_da)
    # 现在严格对齐（应该都 365 天了）
    def_da, vws_da = xr.align(def_da, vws_da, join="exact")
    ti = (def_da * vws_da).astype("float32").transpose("time", "plev", "lat", "lon")

    # --- 关键：把每年的 time 统一成相同的相对索引（0..364） ---
    # 可选：把原来的绝对时间保留下来当辅助坐标（不会参与对齐）
    abs_time = ti.time.values
    ti = ti.assign_coords(time=("time", np.arange(365, dtype=np.int16)))
    ti = ti.assign_coords(time_abs=("time", abs_time))  # 仅做参考，不影响 concat
    return ti

# 1) 构建懒拼接的 5D TI1(year,time,plev,lat,lon)
parts = []
for y in YEARS:
    print(f"[build] TI1 {y}")
    ti_y = open_ti1_year(y).expand_dims(year=[y])
    parts.append(ti_y)

TI1 = xr.concat(parts, dim="year")  # 仍然是 dask-backed，未真正计算
TI1 = TI1.chunk({"year": 1, "time": 30, "plev": 6, "lat": 180, "lon": 360})

# 2) 全时期阈值（对 year,time,lat,lon 取分位数；每个 plev 一个阈值）
ti_sel = TI1.where((TI1.lat >= LAT_MIN) & (TI1.lat <= LAT_MAX), drop=True)
thr_plev = ti_sel.quantile(Q, dim=("year","time","lat","lon"), skipna=True)
if "quantile" in thr_plev.dims:
    thr_plev = thr_plev.squeeze("quantile", drop=True)  # dims: (plev,)

# 3) 结果落盘（只存阈值与逐年百分比，体积非常小）
print('写入中。。。')
thr_plev.name = "threshold"
thr_plev.to_netcdf(f"{OUT_DIR}/era5_TI1_threshold_{int(Q*100)}p_plev.nc")
print('已经写入阈值。。。')
