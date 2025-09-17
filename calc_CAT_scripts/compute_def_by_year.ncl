;===============================================================
; compute_def_total_by_year.ncl  (NCL 6.6.2)
; - 逐年读取 u/v (time, plev, lat, lon)
; - 若纬度为 N->S 则翻转为 S->N
; - 仅保留北半球纬度 0~90°
; - shear_stretch_deform_cfd(u,v,lat,lon,cyclic,opt_ssd) 计算总变形
; - 输出：era5_def_YYYY_50-300hpa.nc，变量名 def(time, plev, lat, lon)
;===============================================================

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"   ; 提供 latFlip、shear_stretch_deform_cfd 等

;-------------------------
; 处理单一年份（定义在 begin 外）
;-------------------------
procedure do_one_year(y:integer, in_dir_uv:string, out_dir:string, OVERWRITE:logical)
local su, sv, out, fu, fv, u, v, time, plev, lat, lon, \
      ntime, nplev, nlat, nlon, def_total, ip, u3d, v3d, ssdList, \
      lat0, lat1, cyclic, opt_ssd, dimNames, dimSizes, dimUnlim, fout, fAtt
begin
  su  = in_dir_uv + "/era5-u-" + sprinti("%0.4i", y) + "-50-300hpa.nc"
  sv  = in_dir_uv + "/era5-v-" + sprinti("%0.4i", y) + "-50-300hpa.nc"
  out = out_dir   + "/era5_def_" + sprinti("%0.4i", y) + "-50-300hpa.nc"

  if (.not.OVERWRITE .and. isfilepresent(out)) then
    print("跳过已存在: " + out)
    return
  end if
  if (.not.isfilepresent(su)) then
    print("❗ 缺少 u 文件: " + su)
    return
  end if
  if (.not.isfilepresent(sv)) then
    print("❗ 缺少 v 文件: " + sv)
    return
  end if

  print("==== 处理年份 " + y + " ====")
  fu = addfile(su, "r")
  fv = addfile(sv, "r")

  ; 变量名兼容
  u = fu->u
  v = fv->v

  ;========= 保证维度顺序为 (time, plev, lat, lon)（按需启用）
  ; u = u(time|:, plev|:, lat|:, lon|:)
  ; v = v(time|:, plev|:, lat|:, lon|:)

  ;========= 若为 N->S，则翻转到 S->N
  lat0 = u&lat(0)
  lat1 = u&lat(dimsizes(u&lat)-1)
  if (lat0 .gt. lat1) then
    u = u(time|:, plev|:, lat|::-1, lon|:)
    v = v(time|:, plev|:, lat|::-1, lon|:)
  end if

  ;========= 仅保留北半球 0~90°
  ; 提取北半球 0–90N
  uN = u(:, :, {0:90}, :)
  vN = v(:, :, {0:90}, :)

  delete(u)
  delete(v)

  u = uN
  v = vN
  delete(uN)
  delete(vN)
  ; 更新坐标与尺寸
  time = u&time
  plev = u&plev
  lat  = u&lat
  lon  = u&lon
  ntime = dimsizes(time)
  nplev = dimsizes(plev)
  nlat  = dimsizes(lat)
  nlon  = dimsizes(lon)

  ; 预分配输出
  def_total = new((/ntime, nplev, nlat, nlon/), float, -9999.0)
  if (.not.ismissing(u@_FillValue)) then
    def_total@_FillValue = u@_FillValue
  end if

  ; shear_stretch_deform_cfd 参数
  cyclic  = True     ; 经度全 0~360 保持周期
  opt_ssd = 0        ; 只要 shear/stretch/total deformation

  ;========= 逐层计算
  do ip = 0, nplev-1
    u3d = u(:, ip, :, :)
    v3d = v(:, ip, :, :)

    ; 6 参数：u,v,lat,lon,cyclic,opt_ssd
    ssdList = shear_stretch_deform_cfd(u3d, v3d, lat, lon, cyclic, opt_ssd)

    ; 文档：ssdList[0]=stretch, [1]=shear, [2]=total deform
    def_total(:, ip, :, :) = tofloat(ssdList[2])

    delete([/u3d, v3d, ssdList/])
  end do

  ;========= 坐标属性
  if (.not. ismissing(plev@units)) then
    ; assumed Pa
  else
    plev@units = "Pa"
  end if
  plev@standard_name = "air_pressure"
  plev@long_name     = "pressure"
  plev@positive      = "down"
  plev@axis          = "Z"

  lat@standard_name  = "latitude"
  lat@long_name      = "latitude"
  lat@units          = "degrees_north"
  lat@axis           = "Y"

  lon@standard_name  = "longitude"
  lon@long_name      = "longitude"
  lon@units          = "degrees_east"
  lon@axis           = "X"

  time@standard_name = "time"
  time@axis          = "T"

    ; ========= 输出（覆盖写） =========
  if (isfilepresent(out)) then
    system("/bin/rm -f " + out)
  end if
  setfileoption("nc", "Format", "NetCDF4Classic")
  setfileoption("nc", "CompressionLevel", 4)

  ; ---- 坐标类型转换：避免 int64 在 Classic 中不被支持 ----
  time_d = todouble(time)    ; 常用 double
  plev_d = todouble(plev)    ; 强制 double（Pa）
  lat_f  = tofloat(lat)      ; 或者 todouble(lat)
  lon_f  = tofloat(lon)      ; 或者 todouble(lon)

  fout = addfile(out, "c")

  dimNames = (/"time","plev","lat","lon"/)
  dimSizes = (/ dimsizes(time_d), dimsizes(plev_d), dimsizes(lat_f), dimsizes(lon_f) /)
  dimUnlim = (/ True, False, False, False /)
  filedimdef(fout, dimNames, dimSizes, dimUnlim)

  ; 用转换后的类型定义坐标变量
  filevardef(fout, "time", typeof(time_d), (/"time"/))
  filevardef(fout, "plev", typeof(plev_d), (/"plev"/))
  filevardef(fout, "lat",  typeof(lat_f),  (/"lat"/))
  filevardef(fout, "lon",  typeof(lon_f),  (/"lon"/))

  ; 写入坐标与属性
  fout->time = time_d
  fout->plev = plev_d
  fout->lat  = lat_f
  fout->lon  = lon_f

  ; 数据变量（总变形），并启用压缩
  setfileoption(fout, "DeflateLevel", 4)
  def_total@long_name   = "total deformation"
  def_total@units       = "s-1"
  def_total@description = "shear_stretch_deform_cfd total deformation over 0–90N"

  filevardef(fout, "def", typeof(def_total), dimNames)
  filevarattdef(fout, "def", def_total)
  fout->def = def_total

  ; 全局属性
  fAtt               = True
  fAtt@Conventions   = "CF-1.6"
  fAtt@title         = "ERA5 total deformation (daily), levels 50–300 hPa, 0–90N"
  fAtt@source        = "shear_stretch_deform_cfd from yearly 4D u/v"
  fAtt@year          = y
  fileattdef(fout, fAtt)

  delete([/u, v, def_total, fu, fv, fout/])
  print("完成：" + out)
end

;-------------------------
; 主驱动
;-------------------------
begin
  in_dir_uv = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_merged"
  out_dir   = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_def"
  system("mkdir -p " + out_dir)

  OVERWRITE = True

  env_year = getenv("YEAR")
  if (.not. ismissing(env_year)) then
    y0 = stringtointeger(env_year)
    do_one_year(y0, in_dir_uv, out_dir, OVERWRITE)
  else
    years = ispan(1979, 2024, 1)
    do iy = 0, dimsizes(years)-1
      do_one_year(years(iy), in_dir_uv, out_dir, OVERWRITE)
    end do
  end if

  print("全部年份处理完成。")
end
