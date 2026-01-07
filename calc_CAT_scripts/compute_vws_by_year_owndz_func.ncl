;===============================================================
; compute_vws_by_year.ncl  (NCL 6.6.2)
; - u/v/gph: (time, plev, lat, lon)，gph 变量名固定为 "gph"
; - 若纬度为 N->S 则在读取时翻为 S->N；随后裁剪 0–90N
; - 垂直差分沿 plev：端点单边、中间中心差分
; - 逐层写出，避免一次性分配超大 4D 中间数组
; - gph 若为势能(m2 s-2)则除 9.80665 转为位势高度(m)
; 输出：era5_vws_YYYY_50-300hpa.nc，变量 vws(time, plev, lat, lon)
;===============================================================

load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"

;-----------------------------
; 处理单一年份（逐层计算、逐层写出）
;-----------------------------
procedure do_one_year(y:integer, dir_uv:string, dir_gph:string, out_dir:string, OVERWRITE:logical)
local su, sv, sg, out, fu, fv, fg, lat1d, lat_desc, \
      u, v, gph, time, plev, lat, lon, nt, np, ny, nx, \
      fout, dimNames, dimSizes, dimUnlim, att, gAtt, fillv, \
      k, du, dv, dz, vws_k, time_d, plev_d, lat_f, lon_f
begin
  su  = dir_uv  + "/era5-u-"    + sprinti("%0.4i", y) + "-50-300hpa.nc"
  sv  = dir_uv  + "/era5-v-"    + sprinti("%0.4i", y) + "-50-300hpa.nc"
  sg  = dir_gph + "/era5-gph-"  + sprinti("%0.4i", y) + "-50-300hpa.nc"
  out = out_dir + "/era5_vws_"  + sprinti("%0.4i", y) + "_50-300hpa.nc"

  if (.not.OVERWRITE .and. isfilepresent(out)) then
    print("跳过已存在: " + out)
    return
  end if
  if (.not.isfilepresent(su)) then
    print("❗ 缺少 u: " + su)  ; 直接返回
    return
  end if
  if (.not.isfilepresent(sv)) then
    print("❗ 缺少 v: " + sv)
    return
  end if
  if (.not.isfilepresent(sg)) then
    print("❗ 缺少 gph: " + sg)
    return
  end if

  print("==== 处理年份 " + y + " ====")

  fu = addfile(su,"r")
  fv = addfile(sv,"r")
  fg = addfile(sg,"r")

  ; 先看纬度顺序
  lat1d   = fu->u&lat
  lat_desc = (lat1d(0) .gt. lat1d(dimsizes(lat1d)-1))

  ; —— 按需翻纬度，并在“读取时”就裁剪 0–90N，减少内存占用 ——
  if (lat_desc) then
    u   = fu->u(time|:364, plev|:, lat|::-1, lon|:)    ; 翻成 S->N
    v   = fv->v(time|:364, plev|:, lat|::-1, lon|:)
    gph = fg->gph(time|:364, plev|:, lat|::-1, lon|:)
  else
    u   = fu->u
    v   = fv->v
    gph = fg->gph
  end if
  printVarSummary(u)
  printVarSummary(v)
  printVarSummary(gph)

  ; gph 单位如为 m2 s-2，转 m
  if (isatt(gph,"units")) then
    if (gph@units.eq."m2 s-2" .or. gph@units.eq."m^2 s^-2") then
      gph = gph / 9.80665
      gph@units = "m"
    end if
  end if

  ; 北半球裁剪（坐标子集）：生成新变量再替换，避免维度冲突
  uN   = u(:, :, {0:90}, :)
  vN   = v(:, :, {0:90}, :)
  gphN = gph(:, :, {0:90}, :)
  delete([/u, v, gph/])
  u=uN
  v=vN
  gph=gphN
  delete([/uN, vN, gphN/])

  time = u&time
  plev = u&plev
  lat  = u&lat
  lon  = u&lon

  nt = dimsizes(time)
  np = dimsizes(plev)
  ny = dimsizes(lat)
  nx = dimsizes(lon)

  ;================= 写文件（先建壳，再逐层写数据） =================
  if (isfilepresent(out)) then
    system("/bin/rm -f " + out)
  end if
  setfileoption("nc","Format","NetCDF4Classic")
  setfileoption("nc","DeflateLevel",4)

  fout = addfile(out, "c")

  ; 坐标类型转换（避免 classic 的 int64 问题）
  time_d = todouble(time)
  plev_d = todouble(plev)
  lat_f  = tofloat(lat)
  lon_f  = tofloat(lon)

  dimNames = (/"time","plev","lat","lon"/)
  dimSizes = (/ nt, np, ny, nx /)
  dimUnlim = (/ True, False, False, False /)
  filedimdef(fout, dimNames, dimSizes, dimUnlim)

  filevardef(fout, "time", typeof(time_d), (/"time"/))
  filevardef(fout, "plev", typeof(plev_d), (/"plev"/))
  filevardef(fout, "lat",  typeof(lat_f),  (/"lat"/))
  filevardef(fout, "lon",  typeof(lon_f),  (/"lon"/))

  att = True
  att@standard_name = "time"
  att@axis          = "T"
  filevarattdef(fout, "time", att)

  att = True
  att@standard_name = "air_pressure"
  att@long_name     = "pressure"
  att@units         = "Pa"
  att@positive      = "down"
  att@axis          = "Z"
  filevarattdef(fout, "plev", att)

  att = True
  att@standard_name = "latitude"
  att@long_name     = "latitude"
  att@units         = "degrees_north"
  att@axis          = "Y"
  filevarattdef(fout, "lat", att)

  att = True
  att@standard_name = "longitude"
  att@long_name     = "longitude"
  att@units         = "degrees_east"
  att@axis          = "X"
  filevarattdef(fout, "lon", att)

  fout->time = time_d
  fout->plev = plev_d
  fout->lat  = lat_f
  fout->lon  = lon_f

  ; 定义 vws 变量（只定义一次）
  filevardef(fout, "vws", typeof(1.0), dimNames)   ; float 类型
  fillv = -9999.0
  att = True
  att@long_name   = "vertical wind shear magnitude"
  att@units       = "s-1"
  att@description = "sqrt((du/dz)^2 + (dv/dz)^2), z is geopotential height (m)"
  att@_FillValue  = fillv
  ;filevarattdef(fout, "vws", att)

  ; k = 0
  du = (u(:,1,:,:) - u(:,0,:,:))
  dv = (v(:,1,:,:) - v(:,0,:,:))
  dz = (gph(:,1,:,:) - gph(:,0,:,:))
  printVarSummary(du)
  printVarSummary(dv)
  printVarSummary(dz)
  vws_k = sqrt( (du/dz)^2 + (dv/dz)^2 )
  vws_k = where(.not.ismissing(dz) .and. dz.ne.0.0, tofloat(vws_k), fillv)
  fout->vws(:,0,:,:) = vws_k
  delete(du)
  delete(dv)
  delete(dz)
  delete(vws_k)

  ; 中间层
  do k = 1, np-2
    du = (u(:,k+1,:,:) - u(:,k-1,:,:))
    dv = (v(:,k+1,:,:) - v(:,k-1,:,:))
    dz = (gph(:,k+1,:,:) - gph(:,k-1,:,:))
    vws_k = sqrt( (du/dz)^2 + (dv/dz)^2 )
    vws_k = where(.not.ismissing(dz) .and. dz.ne.0.0, tofloat(vws_k), fillv)
    fout->vws(:,k,:,:) = vws_k
    delete(du)
    delete(dv)
    delete(dz)
    delete(vws_k)
  end do

  ; k = np-1
  du = (u(:,np-1,:,:) - u(:,np-2,:,:))
  dv = (v(:,np-1,:,:) - v(:,np-2,:,:))
  dz = (gph(:,np-1,:,:) - gph(:,np-2,:,:))
  vws_k = sqrt( (du/dz)^2 + (dv/dz)^2 )
  vws_k = where(.not.ismissing(dz) .and. dz.ne.0.0, tofloat(vws_k), fillv)
  fout->vws(:,np-1,:,:) = vws_k
  delete(du)
  delete(dv)
  delete(dz)
  delete(vws_k)

  ; 全局属性
  gAtt = True
  gAtt@Conventions = "CF-1.6"
  gAtt@title       = "ERA5 vertical wind shear magnitude (daily), 50–300 hPa, 0–90N"
  gAtt@source      = "Finite differences along plev using geopotential height z(m) as vertical coordinate"
  gAtt@year        = y
  fileattdef(fout, gAtt)

  delete(fout)
  print("完成：" + out)
end

;-----------------------------
; 主驱动
;-----------------------------
begin
  dir_uv  = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_merged"
  dir_gph = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_merged"
  out_dir = "/home/dongyl/Work2_2025/ERA5_daily/everyyear_levels_50-300_vws"
  system("mkdir -p " + out_dir)

  OVERWRITE = True

  env_year = getenv("YEAR")
  if (.not. ismissing(env_year)) then
    do_one_year(stringtointeger(env_year), dir_uv, dir_gph, out_dir, OVERWRITE)
  else
    years = ispan(2017, 2017, 1) ;ispan(2017, 2024, 1)
    do iy = 0, dimsizes(years)-1
      do_one_year(years(iy), dir_uv, dir_gph, out_dir, OVERWRITE)
    end do
  end if
end
