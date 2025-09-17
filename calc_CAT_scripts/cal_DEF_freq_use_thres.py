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
    #vws_da = xr.open_dataset(f"{VWS_DIR}/era5_vws_{year}_50-300hpa.nc")["vws"]
    time2 = time.time()
    time2_1 = time2-time1
    print(f"load {year} DEF and VWS, use {time2_1}")

    # 各自多于 365 天就去掉最后一天
    def_da = trim_to_365(def_da)
    # 现在严格对齐（应该都 365 天了）
    def_ = (def_da).astype("float32").transpose("time", "plev", "lat", "lon")
    time3 = time.time()
    time3_2 = time3-time2
    print(f"calculated {year} VWS, use {time3_2}")
    
    return def_

thr_plev = xr.open_dataset("/home/dongyl/Work2_2025/ERA5_daily/era5_DEF_threshold_95p_plev.nc")["def_threshold"]
for year in YEARS:
    def_ = open_ti1_year(year)
    print("DEF done")
    exceed = def_ > thr_plev
    prob_year = (exceed.mean(dim="time", skipna=True) * 100.0).astype("float32")
    prob_year.name = "frequency"

    print('写入中。。。')
    prob_year = prob_year.reset_coords(drop=True)  # 丢掉非索引坐标，如 time_abs

    # 转成 Dataset，便于统一清理坐标与编码
    ds = xr.Dataset({"frequency": prob_year})

    # 1) 删除任何 shape 含 0 的坐标变量；2) 去掉 bounds 属性；3) 清理 encoding（尤其是 _FillValue/missing_value）
    for name in list(ds.coords):
        var = ds.coords[name]
        if 0 in var.shape:
            ds = ds.drop_vars(name)
            continue
        var.attrs.pop("bounds", None)
        var.encoding.clear()

    # 也清一下数据变量的 encoding，防止遗留奇怪的 _FillValue
    ds["frequency"].encoding.clear()
    # 可选：定 dtype、压缩
    ds["frequency"] = ds["frequency"].astype("float32")
    encoding = {"frequency": {"_FillValue": None, "zlib": True, "complevel": 4}}

    # （可选）先 materialize，避免懒计算导致奇怪的编码路径
    ds = ds.compute()

    out = f"{OUT_DIR}/era5_DEF_exceed_pct_{year}_{int(Q * 100)}p.nc"
    ds.to_netcdf(out, encoding=encoding)
    print(f'已经写入{year}频率')
