import numpy as np
import xarray as xr
from scipy import optimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from scipy import stats
import areamean_dhq as areamean
from matplotlib.font_manager import FontProperties

#ta_1951_2099 = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp585/ta/1951-2099/ta_h+ssp585_195101-209912_360x181_yearmean_zonmean_allmodels.nc')
#ua_1951_2099 = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp585/ua/h+ssp585_1951-2099/yearmean_zonmean/ua_ssp585_195101-200012_year_zonmean_all_models_360x181.nc')
ta_1951_2099 = xr.open_dataset('/home/dongyl/Databank/2023summer/historical+ssp245/ta/1951-2099/ta_h+ssp245_195101-209912_360x181_yearmean_zonmean_allmodels.nc')
ua_1951_2099 = xr.open_dataset('')
T_SSP585 = np.mean(ta_1951_2099.ta[:, 109:140, :, :, 0], 1)
T_historical = np.mean(ta_1951_2099.ta[:, 29:60, :, :, 0], 1)
delta_T = T_SSP585 - T_historical
ua_SSP585 = np.mean(ua_1951_2099.ua[:, 109:140, :, :, 0], 1)
ua_historical = np.mean(ua_1951_2099.ua[:, 29:60, :, :, 0], 1)

# Constants
Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization
p = T_SSP585.plev  # Pressure levels


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
beta = np.zeros((15, 19, 181))
u_prime = np.zeros((15, 19, 181))

# Main script execution
for models in range(T_historical.shape[0]):
    # T = np.mean(T_historical,0)
    # delta_T = np.mean(delta_T,0)
    beta[models] = calculate_beta(T_historical[models], delta_T[models], p)
    u_prime[models] = calculate_transformed_u(ua_historical[models], beta[models], p)
delta_VST_u = u_prime - ua_historical
delta_u = ua_SSP585 - ua_historical
delta_VST_u = delta_VST_u.expand_dims(dim='new_dim', axis=3)
delta_u = delta_u.expand_dims(dim='new_dim', axis=3)
delta_VST_u_100_200 = areamean.mask_am(delta_VST_u[:, 11, 111:151, :]) - areamean.mask_am(delta_VST_u[:, 9, 111:151, :])
delta_u_100_200 = areamean.mask_am(delta_u[:, 11, 111:151, :]) - areamean.mask_am(delta_u[:, 9, 111:151, :])
# plt.show()

# 模拟数据
np.random.seed(0)
# 使用 15 个不同的符号表示不同的模式
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X']
models = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CESM2-WACCM', 'CMCC-CM2-SR5', 'FGOALS-f3-L', 'FGOALS-g3', 'INM-CM5-0',
          'KACE-1-0-G', 'KIOST-ESM', 'MPI-ESM1-2-HR', 'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1']
# 绘制散点图
fig, ax = plt.subplots(figsize=(8, 5), dpi=250)
for i in range(15):
    ax.scatter(delta_u_100_200[i], delta_VST_u_100_200[i], marker=markers[i], s=35, label=models[i])
# 线性回归分析
slope, intercept, r_value, p_value, std_err = stats.linregress(delta_u_100_200, delta_VST_u_100_200)

# 绘制回归直线
line = slope * delta_u_100_200 + intercept
ax.plot(delta_u_100_200, line, color='red', linewidth=2)

# 添加图例、标签和标题
ax.legend(fontsize=11, bbox_to_anchor=(1.04, 0.5), loc='center left', frameon=False)
ax.set_xlabel('Simulated Δu$_{100}$-Δu$_{200}$', fontsize=16)
ax.set_ylabel('Estimated Δu$_{100}$-Δu$_{200}$', fontsize=16)

# 在图上标注斜率和显著性
ax.annotate(f'Slope: {slope:.2f}\n$R = {r_value:.2f}$\np<0.01', xy=(0.7, 0.25), xycoords='axes fraction',
            horizontalalignment='left', verticalalignment='top', fontsize=14)
min_val = min(delta_u_100_200.min(), delta_VST_u_100_200.min()) - 0.1
max_val = max(delta_u_100_200.max(), delta_VST_u_100_200.max()) + 0.1
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
