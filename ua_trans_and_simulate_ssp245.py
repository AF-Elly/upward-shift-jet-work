import xarray as xr
import numpy as np
from scipy import optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from scipy import stats
import areamean_dhq as areamean
'''
#ta_1951_2099 = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp585/ta/1951-2099/ta_h+ssp585_195101-209912_360x181_yearmean_zonmean_allmodels.nc')
#ua_1951_2099 = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp585/ua/h+ssp585_1951-2099/yearmean_zonmean/ua_ssp585_195101-200012_year_zonmean_all_models_360x181.nc')
'''
path1 = r'/home/dongyl/Databank/2023summer/reanalysis/ERA5/ua/u_1979-2022_yearmean_zonmean_run5mean_200minus100.nc'
path2 = r'/home/dongyl/Databank/obs-data/MERRA-2/1980-2022/u_MERRA-2_198001-202212_zonmean_yearmean_360x181_run5mean_200minus100.nc'
path3 = r'/home/dongyl/Databank/2023summer/historical+ssp585/ua/h+ssp585_1951-2099/yearmean_zonmean/200minus100/ua_ssp585_1979-2099_year_zonmean_run5mean_allmodels_200minus100.nc'
path4 = r'/home/dongyl/Databank/amip/ua/remapbil_360x180/yearmean_zonmean/ua_amip_197901-201412_yearmean_zonmean_allmodels_200minus100.nc'
'''
path1 = r'/home/dongyl/Databank/obs-data/MERRA-2/ta/yearmean_zonmean/ta_MERRA-2_198001-202212_zonmean_yearmean_181.nc'
path1_ = r'/home/dongyl/Databank/obs-data/MERRA-2/ua/1980-2022/yearmean_zonmean/u_MERRA-2_198001-202212_zonmean_yearmean_181.nc'
path2 = r''
path2_ = r'/home/dongyl/Databank/2023summer/reanalysis/ERA5/ua/yearmean_zonmean/u_1979-2022_yearmean_zonmean_721.nc'
dataset1 = xr.open_dataset(path1)
dataset2 = xr.open_dataset(path2)
dataset1_ = xr.open_dataset(path1_)
dataset2_ = xr.open_dataset(path2_)
T_MERRA2 = dataset1.t[:,0,:,:]
ua_MERRA2 = dataset1_.U[:,0,:,:]
T_ERA5 = dataset2.t[:,0,:,:]
ua_ERA5 = dataset2_.u[:,0,:,:]
#amip=dataset4.ua[:,:,0,:,:]
T_MERRA2_areamean = areamean.mask_am(T_MERRA2[:, 110:150, :])
T_ERA5_areamean = areamean.mask_am(T_ERA5[:, 440:601, :])
#subplot_hist_allmodels = areamean.mask_am4D(h_ssp585[:, 28:60, 97:148, :])
#subplot_amip_allmodels = areamean.mask_am4D(amip[:, :, 110:150, :])
# 创建时间数组
time_ERA5 = np.arange(1981, 2021)  # 1981-2020
time_MERRA2 = np.arange(1982, 2021)  # 1982-2020
time_hist = np.arange(1981, 2013)  # 1981-2097
time_amip = np.arange(1981, 2013)  # 1981-2097
# 计算每个模式的趋势和显著性
years = np.arange(1981, 2013)
trends_amip = []
p_values_amip = []
trends_hist = []
p_values_hist = []
trends_ERA5 = []
p_values_ERA5 = []
trends_MERRA2 = []
p_values_MERRA2 = []

for i in range(subplot_amip_allmodels.shape[0]):
    slope, _, _, p_value, _ = stats.linregress(years, -subplot_amip_allmodels[i, :])
    trends_amip.append(slope)
    p_values_amip.append(p_value)
for i in range(subplot_hist_allmodels.shape[0]):
    slope, _, _, p_value, _ = stats.linregress(years, -subplot_hist_allmodels[i, :])
    trends_hist.append(slope)
    p_values_hist.append(p_value)
slope, _, _, p_value, _ = stats.linregress(years, -subplot_amip_allmodels.mean(axis=0))
trends_amip.append(slope)
p_values_amip.append(p_value)

slope, _, _, p_value, _ = stats.linregress(years, -subplot_hist_allmodels.mean(axis=0))
trends_hist.append(slope)
p_values_hist.append(p_value)

slope, _, _, p_value, _ = stats.linregress(time_ERA5, T_ERA5_areamean)
trends_ERA5.append(slope)
p_values_ERA5.append(p_value)

slope, _, _, p_value, _ = stats.linregress(time_MERRA2, T_MERRA2_areamean)
trends_MERRA2.append(slope)
p_values_MERRA2.append(p_value)
####用trend计算delta_T
delta_T_ERA5 = trends_ERA5*80
delta_T_MERRA2 = trends_MERRA2*80
######################################################################
'''
ta_1951_2099 = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp245/ta/1951-2099/both_in_ua/ta_h+ssp245_195101-209912_360x181_yearmean_zonmean_allmodels.nc')
ua_1951_2099 = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp245/ua/h+ssp245_1951-2099/yearmean_zonmean/both_in_ta/ua_h+ssp245_360x181_yearmean_zonmean_allmodels.nc')
T_SSP245 = np.mean(ta_1951_2099.ta[:, 109:140, :, :, 0], 1)
T_historical = np.mean(ta_1951_2099.ta[:, 29:60, :, :, 0], 1)
delta_T = T_SSP245 - T_historical
ua_SSP245 = np.mean(ua_1951_2099.ua[:, 109:140, :, :, 0], 1)
ua_historical = np.mean(ua_1951_2099.ua[:, 29:60, :, :, 0], 1)

# Constants
Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization
p = T_SSP245.plev  # Pressure levels


def calculate_beta(T, delta_T, p):
    # T_fine = interpolate_to_fine(T, p, p_fine)
    # delta_T_fine = interpolate_to_fine(delta_T, p, p_fine)
    # Calculate es, dT_dp, and dT_des as before
    dT_dp = np.gradient(T, p, axis=0)

    # Calculate beta for each time, level, and latitude
    def func(beta, delta_T, dT_dp, T, level):
        # p_adjusted = np.tile(p_fine, (181, 1)).T
        return delta_T - (beta - 1) * (p[level] * dT_dp - Rv * T ** 2 / Lv)

    # Solve for beta
    beta_init = 1.15
    beta = np.empty_like(T)
    for lat in range(T.shape[1]):
        # for level in range(p.size):
        #下式中的5代表500hPa的对应序号
        beta[:, lat] = optimize.newton(func, beta_init, args=(delta_T[5, lat], dT_dp[5, lat], T[5, lat], 5))
    # Return the calculated beta array
    return beta


def calculate_transformed_u(u, beta, p):
    # Calculate u_prime as before using interpolated u and beta
    # Calculate transformed u'
    u_prime = np.empty_like(u)
    for lat in range(u.shape[1]):
        # for level in range(p.size):
        transformed_p = beta[:, lat] * p
        u_prime[::-1, lat] = np.interp(transformed_p[::-1], p[::-1], u[::-1, lat])
    # Return the calculated u_prime array
    return u_prime


level = T_historical.plev / 100
level_label = list(map(int, level.data.tolist()))
beta = np.zeros((T_historical.shape[0], 19, 181))
u_prime = np.zeros((T_historical.shape[0], 19, 181))

# Main script execution
for models in range(T_historical.shape[0]):
    # T = np.mean(T_historical,0)
    # delta_T = np.mean(delta_T,0)
    beta[models] = calculate_beta(T_historical[models], delta_T[models], p)
    u_prime[models] = calculate_transformed_u(ua_historical[models], beta[models], p)
delta_VST_u = u_prime - ua_historical
delta_u = ua_SSP245 - ua_historical
delta_VST_u = delta_VST_u.expand_dims(dim='new_dim', axis=3)
delta_u = delta_u.expand_dims(dim='new_dim', axis=3)
delta_VST_u_100_200 = areamean.mask_am(delta_VST_u[:, 11, 50:70, :]) - areamean.mask_am(delta_VST_u[:, 9, 50:70, :])
delta_u_100_200 = areamean.mask_am(delta_u[:, 11, 50:70, :]) - areamean.mask_am(delta_u[:, 9, 50:70, :])
delta_VST_u_100_200_mme = np.mean(areamean.mask_am(delta_VST_u[:, 11, 50:70, :]) - areamean.mask_am(delta_VST_u[:, 9, 50:70, :]),0)
delta_u_100_200_mme = np.mean(areamean.mask_am(delta_u[:, 11, 50:70, :]) - areamean.mask_am(delta_u[:, 9, 50:70, :]),0)
# plt.show()

# 模拟数据
np.random.seed(0)
# 使用 15 个不同的符号表示不同的模式
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '1', 'h', 'H', 'D', 'd', 'P', 'X','x']
models = ['ACCESS-CM2', 'BCC-CSM2-MR',  'CAS-ESM2-0', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'FGOALS-f3-L', 'FGOALS-g3', 'INM-CM5-0',
          'KACE-1-0-G', 'KIOST-ESM', 'MIROC6', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1','MME']
fig, ax = plt.subplots(figsize=(8, 5), dpi=250)
# 线性回归分析
slope, intercept, r_value, p_value, std_err = stats.linregress(delta_u_100_200, delta_VST_u_100_200)
# 绘制回归直线
line = slope * delta_u_100_200 + intercept
ax.plot(delta_u_100_200, line, color='red', linewidth=2)
'''
ax.axvspan(np.mean(np.array(trends_amip)*80)-np.std(np.array(trends_amip)*80),np.mean(np.array(trends_amip)*80)+np.std(np.array(trends_amip)*80),
           color='#B3BC95', alpha=0.5,zorder=0,label='AMIP')#amip
ax.axvspan(np.mean(np.array(trends_hist)*80)-np.std(np.array(trends_hist)*80),np.mean(np.array(trends_hist)*80)+np.std(np.array(trends_hist)*80),
           color='#ADBED6', alpha=0.5,zorder=1,label='hist')#hist
ax.axvline(x=np.array(trends_ERA5)*80,color='#E57259',alpha=0.6,label='ERA5')#ERA5
ax.axvline(x=np.array(trends_MERRA2)*80,color='#F2B9AC',alpha=1,label='MERRA2')#MERRA2
'''
plt.legend()
# 绘制散点图
for i in range(T_historical.shape[0]):
    ax.scatter(delta_u_100_200[i], delta_VST_u_100_200[i], marker=markers[i], s=35, label=models[i])
ax.scatter(delta_u_100_200_mme,delta_VST_u_100_200_mme,marker='*', s=200,color='k',zorder=20,label=models[-1])
# 添加图例、标签和标题
ax.legend(fontsize=11, bbox_to_anchor=(1.04, 0.5), loc='center left', frameon=False)
ax.set_xlabel('Simulated Δu$_{100}$-Δu$_{200}$', fontsize=12)
ax.set_ylabel('Estimated Δu$_{100}$-Δu$_{200}$', fontsize=12)
#plt.text(0.95, 1.05, 'historical SSP245', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=14)
# 在图上标注斜率和显著性
ax.annotate(f'Slope: {slope:.2f}\n$R = {r_value:.2f}$\np<0.01', xy=(0.7, 0.25), xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top', fontsize=12)
min_val = min(delta_u_100_200.min(), delta_VST_u_100_200.min()) - 0.1
max_val = max(delta_u_100_200.max(), delta_VST_u_100_200.max()) + 0.1
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', linewidth=1)
# 增大坐标轴标签和刻度的字体大小
ax.tick_params(axis='both', which='major', labelsize=12)
ax.set_title('Based on T', loc='left', fontsize=12)
ax.set_title('SSP2-4.5 (20°-40°S)', loc='right', fontsize=12)
plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.grid(True)
# 展示图像
plt.show()