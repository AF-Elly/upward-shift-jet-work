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

T_ERA5_1979_2022 = xr.open_dataset('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/ERA5/ta/ta_ERA5_1979-2022_yearmean_zonmean_721.nc')
U_ERA5_1979_2022 = xr.open_dataset('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/ERA5/ua/yearmean_zonmean/u_1979-2022_yearmean_zonmean_721.nc')
T_MERRA2_1979_2022 = xr.open_dataset('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/MERRA-2/ta/yearmean_zonmean/ta_MERRA-2_198001-202212_zonmean_yearmean_181.nc')
U_MERRA2_1979_2022 = xr.open_dataset('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/MERRA-2/ua/1980-2022/yearmean_zonmean/u_MERRA-2_198001-202212_zonmean_yearmean_181.nc')

ua_ERA5 = U_ERA5_1979_2022.u[:,::-1, :, 0] #变量顺序：year,level,lat,lon
T_ERA5 = T_ERA5_1979_2022.t[:,::-1, :, 0]
T_MERRA2 = T_MERRA2_1979_2022.T[:,:,:,0]
ua_MERRA2 = U_MERRA2_1979_2022.U[:,:,:,0]
years_ERA5 = np.arange(1979, 2015)
years_MERRA2 = np.arange(1980, 2015)
trends_ERA5=[]
p_values_ERA5=[]
trends_MERRA2=[]
p_values_MERRA2=[]

#########################Estimated u100-u200############################################
for i in range(T_ERA5.shape[2]):
    slope, _, _, p_value, _ = stats.linregress(years_ERA5, T_ERA5_1979_2022.t[:36,13,i,0])
    trends_ERA5.append(slope)
    p_values_ERA5.append(p_value)
for i in range(T_MERRA2.shape[2]):
    slope, _, _, p_value, _ = stats.linregress(years_MERRA2, T_MERRA2_1979_2022.T[:35,16,i,0])
    trends_MERRA2.append(slope)
    p_values_MERRA2.append(p_value)
delta_T_ERA5 = np.array(trends_ERA5)*80
delta_T_MERRA2 = np.array(trends_MERRA2)*80

Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization
p_MERRA2 = T_MERRA2.lev  # Pressure levels,unit:Pa
p_ERA5 = T_ERA5.level

def calculate_beta(T, delta_T, p,sequence_500hpa):
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
        beta[:, lat] = optimize.newton(func, beta_init, args=(delta_T[lat], dT_dp[sequence_500hpa, lat], T[sequence_500hpa, lat], sequence_500hpa))
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

beta_ERA5 = calculate_beta(np.mean(T_ERA5,0), delta_T_ERA5, p_ERA5,5)
u_prime_ERA5 = calculate_transformed_u(np.mean(ua_ERA5,0), beta_ERA5, p_ERA5)
delta_VST_u_ERA5 = (np.mean(u_prime_ERA5[11,200:321]) - np.mean(ua_ERA5[:,11,200:321]))-(np.mean(u_prime_ERA5[9,200:321]) - np.mean(ua_ERA5[:,9,200:321]))

beta_MERRA2 = calculate_beta(np.mean(T_MERRA2,0), delta_T_MERRA2, p_MERRA2,16)
u_prime_MERRA2 = calculate_transformed_u(np.mean(ua_MERRA2,0), beta_MERRA2, p_MERRA2)
delta_VST_u_MERRA2 = (np.mean(u_prime_MERRA2[24,110:151]) - np.mean(ua_MERRA2[:,24,110:151]))-(np.mean(u_prime_MERRA2[22,110:151]) - np.mean(ua_MERRA2[:,22,110:151]))
#👆注意纬度范围
########################################################################

####################Simulated u100-u200################################
u_100_200_ERA5_his_run5mean = xr.open_dataset('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/ERA5/ua/u_1979-2022_yearmean_zonmean_run5mean_200minus100.nc').u[:,0, :, :]
u_100_200_MERRA2_his_run5mean = xr.open_dataset('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/MERRA-2/ua/1980-2022/u_MERRA-2_198001-202212_zonmean_yearmean_360x181_run5mean_200minus100.nc').U[:,0, :, :]
u_100_200_ERA5_areamean = areamean.mask_am(u_100_200_ERA5_his_run5mean[:34, 200:321, :])
u_100_200_MERRA2_areamean = areamean.mask_am(u_100_200_MERRA2_his_run5mean[:33, 111:151, :])####注意纬度范围
slope_ERA5, _, _, p_values_ERA5, _ = stats.linregress(np.arange(1981,2015), -u_100_200_ERA5_areamean)
slope_MERRA2, _, _, p_values_MERRA2, _ = stats.linregress(np.arange(1982,2015), -u_100_200_MERRA2_areamean)
delta_u_ERA5 = np.array(slope_ERA5)*80
delta_u_MERRA2 = np.array(slope_MERRA2)*80
#############################################################################
##################################ssp245 Estimated u100-u200#################################
ta_1951_2099 = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp245/ta/1951-2099/both_in_ua/ta_h+ssp245_195101-209912_360x181_yearmean_zonmean_allmodels.nc')
ua_1951_2099 = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp245/ua/h+ssp245_1951-2099/yearmean_zonmean/both_in_ta/ua_h+ssp245_360x181_yearmean_zonmean_allmodels.nc')
T_hist = ta_1951_2099.ta[:, 28:64, :, :, 0]
ua_hist = ua_1951_2099.ua[:, 28:64, :, :, 0]
years_hist = np.arange(1979, 2015)
trends_hist=np.zeros((T_hist.shape[0],T_hist.shape[3]))
p_values_hist=np.zeros((T_hist.shape[0],T_hist.shape[3]))
delta_VST_u_hist_100_200=np.zeros(T_hist.shape[0])
beta_hist = np.zeros((T_hist.shape[0], 19, 181))
u_prime_hist = np.zeros((T_hist.shape[0], 19, 181))
p = T_hist.plev
for model in range(T_hist.shape[0]):
    for i in range(T_hist.shape[3]):
        slope, _, _, p_value, _ = stats.linregress(years_hist, T_hist[model,:,5,i])
        trends_hist[model,i] = slope
        p_values_hist[model,i] = p_value
delta_T_hist = trends_hist*80
# Main script execution
for models in range(T_hist.shape[0]):
    # T = np.mean(T_historical,0)
    # delta_T = np.mean(delta_T,0)
    beta_hist[models] = calculate_beta(np.mean(T_hist[models],0), delta_T_hist[models],p,5)
    u_prime_hist[models] = calculate_transformed_u(np.mean(ua_hist[models],0), beta_hist[models],p)
    delta_VST_u_hist_100_200[models] = (np.mean(u_prime_hist[models,11,111:151]) - np.mean(ua_hist[models,:,11,111:151]))-(np.mean(u_prime_hist[models,9,111:151]) - np.mean(ua_hist[models,:,9,111:151]))

###########################ssp245 Simulated u100-u200#######################################
u_100_200_hist_run5mean = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp245/ua/h+ssp245_1951-2099/yearmean_zonmean/run5mean/200minus100/ua_h+ssp245_yearmean_zonmean_run5mean_200minus100_allmodels_for_simulate.nc').ua[:,:,0,:,:]
u_100_200_hist_areamean = areamean.mask_am4D(u_100_200_hist_run5mean[:,28:60,88:121, :])####注意纬度范围
delta_u_hist_100_200 = np.zeros(T_hist.shape[0])
delta_u_hist_100_200_p_values = np.zeros(T_hist.shape[0])
for models in range(T_hist.shape[0]):
    slope_hist, _, _, p_values_hist, _ = stats.linregress(np.arange(1981,2013), -u_100_200_hist_areamean[models])
    print(slope_hist)
    delta_u_hist_100_200[models] = np.array(slope_hist)*80
    delta_u_hist_100_200_p_values[models] = p_values_hist
#####################################################################################
#################################绘图##########################################
fig, ax = plt.subplots(figsize=(8, 5), dpi=250)
ax.scatter(delta_u_MERRA2, delta_VST_u_MERRA2, marker='o', s=100, color='r',label='MERRA2')
ax.scatter(delta_u_ERA5, delta_VST_u_ERA5, marker='o', s=100, color='b', label='ERA5')
# 模拟数据
np.random.seed(0)

# 使用 15 个不同的符号表示不同的模式
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '1', 'h', 'H', 'D', 'd', 'P', 'X','x']
models = ['ACCESS-CM2', 'BCC-CSM2-MR',  'CAS-ESM2-0', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'FGOALS-f3-L', 'FGOALS-g3', 'INM-CM5-0',
          'KACE-1-0-G', 'KIOST-ESM', 'MIROC6', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1','MME']
# 线性回归分析
slope, intercept, r_value, p_value, std_err = stats.linregress(delta_u_hist_100_200, delta_VST_u_hist_100_200)
# 绘制回归直线
line = slope * delta_u_hist_100_200 + intercept
ax.plot(delta_u_hist_100_200, line, color='red', linewidth=1)
plt.legend()
# 绘制散点图
for i in range(T_hist.shape[0]):
    ax.scatter(delta_u_hist_100_200[i], delta_VST_u_hist_100_200[i], marker='<', s=35, label=models[i])
ax.scatter(np.mean(delta_u_hist_100_200),np.mean(delta_VST_u_hist_100_200), marker='*', s=200,color='k',zorder=20,label=models[-1])
# 添加图例、标签和标题
ax.legend(fontsize=11, bbox_to_anchor=(1.04, 0.5), loc='center left', frameon=False)
ax.set_xlabel('Simulated Δu$_{100}$-Δu$_{200}$', fontsize=16)
ax.set_ylabel('Estimated Δu$_{100}$-Δu$_{200}$', fontsize=16)
#plt.text(0.95, 1.05, 'historical SSP245', horizontalalignment='right', verticalalignment='top', transform=ax.transAxes, fontsize=14)
# 在图上标注斜率和显著性
ax.annotate(f'Slope: {slope:.2f}\n$R = {r_value:.2f}$\np=0.105', xy=(0.7, 0.25), xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top', fontsize=14)
min_val = min(delta_u_hist_100_200.min(), delta_VST_u_hist_100_200.min()) - 0.1
max_val = max(delta_u_hist_100_200.max(), delta_VST_u_hist_100_200.max()) + 0.1
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)
ax.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', linewidth=1)
# 增大坐标轴标签和刻度的字体大小
ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.subplots_adjust(right=0.7)
plt.grid(True)
# 展示图像
plt.show()
'''
fig2, ax = plt.subplots(figsize=(8, 5), dpi=250)
for i in range(16):
    ax.plot(u_100_200_hist_areamean[i])
plt.show()
trends_hist
'''