#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute pooled percentile thresholds (per plev) for DEF and VWS, ERA5 1979–2024.
- Build 5D stacks: (year, time[0..364], plev, lat, lon)
- Pool over year×time×lat×lon within 20–60°N, then take quantile per plev
- Save tiny NetCDFs: era5_DEF_threshold_99p_plev.nc, era5_VWS_threshold_99p_plev.nc
"""

import os
import numpy as np
import xarray as xr

# ====== I/O ======
DEF_DIR = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_def"
VWS_DIR = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_vws"
OUT_DIR = "/home/dongyl/Work2_2025/ERA5_daily"
os.makedirs(OUT_DIR, exist_ok=True)

# ====== Config ======
YEARS   = list(range(1979, 2024 + 1))
Q       = 0.95                      # 99% 分位；需要 95% 就改成 0.95
LAT_MIN, LAT_MAX = 20, 60
CHUNKS  = {"time": 30, "lat": 180, "lon": 360}  # 按你现有存储块来

xr.set_options(keep_attrs=True)

def _trim_to_365(da: xr.DataArray) -> xr.DataArray:
    """排序 → 去除重复时刻 → 截到前 365 天（闰年/异常年）"""
    da = da.sortby("time")
    # 如果存在重复的 time（偶见 ERA5 拼接/转码），去重保留首次出现
    t = da["time"].values
    uniq, idx = np.unique(t, return_index=True)
    if len(uniq) != len(t):
        da = da.isel(time=np.sort(idx))
    # 截到 365 天
    if da.sizes.get("time", 0) > 365:
        da = da.isel(time=slice(0, 365))
    return da

def _open_year_var(year: int, kind: str) -> xr.DataArray:
    """
    kind ∈ {"DEF","VWS"}
    统一输出 dims: (time, plev, lat, lon)，并将 time 设为 0..364，同时保留 time_abs 辅助坐标。
    """
    if kind == "DEF":
        path = f"{DEF_DIR}/era5_def_{year}-50-300hpa.nc"
        var  = "def"
    elif kind == "VWS":
        path = f"{VWS_DIR}/era5_vws_{year}_50-300hpa.nc"
        var  = "vws"
    else:
        raise ValueError("kind must be 'DEF' or 'VWS'")

    da = xr.open_dataset(path, chunks=CHUNKS)[var]
    da = _trim_to_365(da)
    abs_time = da.time.values
    da = da.transpose("time", "plev", "lat", "lon")
    # 标准化“年内日序”索引，避免跨年的绝对时间导致 concat 错配
    da = da.assign_coords(
        time=("time", np.arange(365, dtype=np.int16)),
        time_abs=("time", abs_time)
    )
    return da

def _build_stack(kind: str) -> xr.DataArray:
    """拼接 (year, time, plev, lat, lon) 大数组，保持 dask 懒计算。"""
    parts = []
    for y in YEARS:
        print(f"[build] {kind} {y}")
        parts.append(_open_year_var(y, kind).expand_dims(year=[y]))
    da = xr.concat(parts, dim="year")
    return da.chunk({"year": 1, "time": 30, "plev": 6, "lat": 180, "lon": 360})

def _compute_threshold(kind: str) -> xr.DataArray:
    """在 20–60°N 上池化 year×time×lat×lon，按 plev 求 Q 分位阈值。"""
    da = _build_stack(kind)
    da_sel = da.where((da.lat >= LAT_MIN) & (da.lat <= LAT_MAX), drop=True)
    thr = da_sel.quantile(Q, dim=("year", "time", "lat", "lon"), skipna=True)
    if "quantile" in thr.dims:
        thr = thr.squeeze("quantile", drop=True)  # -> dims: (plev,)
    units = da.attrs.get("units", "")
    thr.name = f"{kind.lower()}_threshold"
    thr.attrs.update({
        "long_name": f"{kind} {int(Q*100)}th percentile threshold (pooled over 1979–2024, {LAT_MIN}–{LAT_MAX}°N)",
        "units": units
    })
    return thr

def main():
    kind = "VWS" #("DEF", "VWS")
    print(f"\n=== {kind}: computing {int(Q*100)}th percentile threshold ===")
    thr = _compute_threshold(kind).compute()  # 先算出来，写盘更快更稳
    out_path = f"{OUT_DIR}/era5_{kind}_threshold_{int(Q*100)}p_plev.nc"
    thr.to_netcdf(out_path)
    print(f"[OK] Saved → {out_path}")

if __name__ == "__main__":
    main()
