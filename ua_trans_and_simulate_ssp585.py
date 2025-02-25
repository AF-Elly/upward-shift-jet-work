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

T_hist = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_1980-2010_zonmean_historical_289x145.nc').hus[:,0,:,:,0]
T_future = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_2060-2090_zonmean_future_289x145.nc').hus[:,0,:,:,0]
ua_hist =xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_1980-2010_zonmean_historical_289x145.nc').ua[:,0,:,:,0]
ua_future = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_2060-2090_zonmean_future_289x145.nc').ua[:,0,:,:,0]

delta_T = T_future - T_hist

# Constants
Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization
p = T_hist.plev  # Pressure levels


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


level = T_hist.plev / 100
level_label = list(map(int, level.data.tolist()))
beta = np.zeros(ua_hist.shape)
u_prime = np.zeros(ua_hist.shape)

# Main script execution
for models in range(T_hist.shape[0]):
    # T = np.mean(T_historical,0)
    # delta_T = np.mean(delta_T,0)
    beta[models] = calculate_beta(T_hist[models], delta_T[models], p)
    u_prime[models] = calculate_transformed_u(ua_hist[models], beta[models], p)
delta_VST_u = u_prime - ua_hist
delta_u = ua_future - ua_hist
delta_VST_u = delta_VST_u.expand_dims(dim='new_dim', axis=3)
delta_u = delta_u.expand_dims(dim='new_dim', axis=3)
delta_VST_u_100_200 = [(areamean.mask_am(delta_VST_u[:, 11, 88:104, :]) - areamean.mask_am(delta_VST_u[:, 9, 88:104, :])),
                        (areamean.mask_am(delta_VST_u[:, 11, 40:56, :]) - areamean.mask_am(delta_VST_u[:, 9, 40:56, :])),
                       (areamean.mask_am(delta_VST_u[:, 11, 88:104, :]) - areamean.mask_am(delta_VST_u[:, 9, 88:104, :])
                        +areamean.mask_am(delta_VST_u[:, 11, 40:56, :]) - areamean.mask_am(delta_VST_u[:, 9, 40:56, :]))/2]
delta_u_100_200 = [(areamean.mask_am(delta_u[:, 11, 88:104, :]) - areamean.mask_am(delta_u[:, 9, 88:104, :])),
    (areamean.mask_am(delta_u[:, 11, 40:56, :]) - areamean.mask_am(delta_u[:, 9, 40:56, :])),
    (areamean.mask_am(delta_u[:, 11, 88:104, :]) - areamean.mask_am(delta_u[:, 9, 88:104, :])
     +areamean.mask_am(delta_u[:, 11, 40:56, :]) - areamean.mask_am(delta_u[:, 9, 40:56, :]))/2]
delta_VST_u_100_200_mme = [np.mean(delta_VST_u_100_200[0],0),np.mean(delta_VST_u_100_200[1],0),np.mean(delta_VST_u_100_200[2],0)]
delta_u_100_200_mme = [np.mean(delta_u_100_200[0],0),np.mean(delta_u_100_200[1],0),np.mean(delta_u_100_200[2],0)]
# plt.show()

# 模拟数据
np.random.seed(0)
# 使用 15 个不同的符号表示不同的模式
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '1', 'h', 'H', 'D', 'd', 'P', 'X']
models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'CanESM5','CAS-ESM2-0','CESM2-WACCM', 'CMCC-CM2-SR5',
          'CMCC-ESM2', 'EC-Earth3', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR','FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0',
          'GFDL-ESM4', 'IITM-ESM', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR',
          'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1','MME']
title=['SSP5-8.5 (20°-40° N)','SSP5-8.5 (20°-40° S)','SSP5-8.5 (20°-40° N & S)']
fig = plt.figure(figsize=(8, 12), dpi=250)
axes=[[.15,.71,.45,.25], [.15,.38,.45,.25],[.15,.05,.45,.25]]
for i in range(3):
    ax = fig.add_axes(axes[i])
    # 线性回归分析
    slope, intercept, r_value, p_value, std_err = stats.linregress(delta_u_100_200[i], delta_VST_u_100_200[i])
    # 绘制回归直线
    line = slope * delta_u_100_200[i] + intercept
    ax.plot(delta_u_100_200[i], line, color='red', linewidth=2)

    # 绘制散点图
    for j in range(ua_hist.shape[0]):
        ax.scatter(delta_u_100_200[i][j], delta_VST_u_100_200[i][j], s=35, marker=markers[j % 15], label=models[j])
    ax.scatter(delta_u_100_200_mme[i], delta_VST_u_100_200_mme[i], marker='*', s=200, color='k', zorder=20,
               label=models[-1])
    # 添加图例、标签和标题
    ax.set_xlabel('Simulated Δu$_{100}$-Δu$_{200}$', fontsize=12)
    ax.set_ylabel('Estimated Δu$_{100}$-Δu$_{200}$', fontsize=12)

    # 在图上标注斜率和显著性
    ax.annotate(f'Slope: {slope:.2f}\n$R = {r_value:.2f}$\np<0.01', xy=(0.7, 0.25), xycoords='axes fraction',
                horizontalalignment='left', verticalalignment='top', fontsize=12)
    # min_val = min(delta_u_100_200[i].min(), delta_VST_u_100_200[i].min()) - 0.1
    # max_val = max(delta_u_100_200[i].max(), delta_VST_u_100_200[i].max()) + 0.1
    ax.set_xlim(1.5, 6.5)
    ax.set_ylim(1.5, 6.5)
    # ax.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', linewidth=1)
    ax.plot([1.5, 6.5], [1.5, 6.5], color='grey', linestyle='--', linewidth=1)
    # 增大坐标轴标签和刻度的字体大小
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_title('Based on T', loc='left', fontsize=12)
    ax.set_title(title[i], loc='right', fontsize=12)
    ax.grid(True)
# 增大坐标轴标签和刻度的字体大小
plt.legend(fontsize=10, bbox_to_anchor=(1.6, 1.7), loc='right', frameon=False)
#bbox_to_anchor=(1.04, 0.5)
plt.tight_layout()
plt.subplots_adjust(right=0.7)
# 展示图像
plt.show()

