import numpy as np
import xarray as xr
from scipy import optimize
import copy
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
from scipy import stats
import areamean_dhq as areamean
from scipy.stats import linregress
import dyl_function_slope as dyl
plt.rcParams['font.family'] = 'Arial'

h_ssp245N = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ua_h+ssp245_all_models_1979-2099_N_100_200hPa.nc').ua[:,:,:,0,0]
h_ssp245S = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ua_h+ssp245_all_models_1979-2099_N_100_200hPa.nc').ua[:,:,:,0,0]
h_ssp245_100minus200N=h_ssp245N[:,:,1]-h_ssp245N[:,:,0]
h_ssp245_100minus200S=h_ssp245S[:,:,1]-h_ssp245S[:,:,0]
trends_ssp245N, p_values_ssp245N = dyl.calculate_trend(h_ssp245_100minus200N[:,:44])
mean_trend_ssp245N,_,_, mean_p_value_ssp245N,_ = linregress(np.arange(44),np.mean(h_ssp245_100minus200N[:,:44],0))
trends_ssp245S, p_values_ssp245S = dyl.calculate_trend(h_ssp245_100minus200S[:,:44])
mean_trend_ssp245S,_,_, mean_p_value_ssp245S,_ = linregress(np.arange(44),np.mean(h_ssp245_100minus200S[:,:44],0))

h_ssp245_DJF = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ua_h+ssp245_all_models_1979-2099_zonmean_289x145DJF.nc').ua
h_ssp245_JJA = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ua_h+ssp245_all_models_1979-2099_zonmean_289x145JJA.nc').ua
h_ssp245_100minus200N_DJF=areamean.mask_am4D(h_ssp245_DJF[:,:,11,89:105])-areamean.mask_am4D(h_ssp245_DJF[:,:,9,89:105])
h_ssp245_100minus200N_JJA=areamean.mask_am4D(h_ssp245_JJA[:,:,11,89:105])-areamean.mask_am4D(h_ssp245_JJA[:,:,9,89:105])
h_ssp245_100minus200S_DJF=areamean.mask_am4D(h_ssp245_DJF[:,:,11,40:56])-areamean.mask_am4D(h_ssp245_DJF[:,:,9,40:56])
h_ssp245_100minus200S_JJA=areamean.mask_am4D(h_ssp245_JJA[:,:,11,40:56])-areamean.mask_am4D(h_ssp245_JJA[:,:,9,40:56])
trends_ssp245N_DJF, p_values_ssp245N_DJF = dyl.calculate_trend(h_ssp245_100minus200N_DJF[:,1:44])
mean_trend_ssp245N_DJF,_,_, mean_p_value_ssp245N_DJF,_ = linregress(np.arange(43),np.mean(h_ssp245_100minus200N_DJF[:,1:44],0))
trends_ssp245N_JJA, p_values_ssp245N_JJA = dyl.calculate_trend(h_ssp245_100minus200N_JJA[:,:44])
mean_trend_ssp245N_JJA,_,_, mean_p_value_ssp245N_JJA,_ = linregress(np.arange(44),np.mean(h_ssp245_100minus200N_JJA[:,:44],0))
trends_ssp245S_DJF, p_values_ssp245S_DJF = dyl.calculate_trend(h_ssp245_100minus200S_DJF[:,:44])
mean_trend_ssp245S_DJF,_,_, mean_p_value_ssp245S_DJF,_ = linregress(np.arange(44),np.mean(h_ssp245_100minus200S_DJF[:,:44],0))
trends_ssp245S_JJA, p_values_ssp245S_JJA = dyl.calculate_trend(h_ssp245_100minus200S_JJA[:,:44])
mean_trend_ssp245S_JJA,_,_, mean_p_value_ssp245S_JJA,_ = linregress(np.arange(44),np.mean(h_ssp245_100minus200S_JJA[:,:44],0))


ts = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ts_h+ssp245_all_models_1979-2022_zonmean_289x145.nc').ts
ts_DJF = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ts_h+ssp245_all_models_1979-2022_zonmean_289x145.nc').ts[:,1:43]
ts_JJA = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp245/ts_h+ssp245_all_models_1979-2022_zonmean_289x145.nc').ts

gmt = areamean.mask_am4D(ts)
gmt_mme = np.nanmean(gmt,axis=0)
trends_gmt, p_values_gmt = dyl.calculate_trend(gmt)
mean_trend_gmt,_,_, mean_p_value_gmt,_ = linregress(np.arange(44),gmt_mme)

gmt_DJF = areamean.mask_am4D(ts_DJF)
gmt_mme_DJF = np.nanmean(gmt_DJF,axis=0)
trends_gmt_DJF, p_values_gmt_DJF = dyl.calculate_trend(gmt_DJF)
mean_trend_gmt_DJF,_,_, mean_p_value_gmt_DJF,_ = linregress(np.arange(42),gmt_mme_DJF)

gmt_JJA = areamean.mask_am4D(ts_JJA)
gmt_mme_JJA = np.nanmean(gmt_JJA,axis=0)
trends_gmt_JJA, p_values_gmt_JJA = dyl.calculate_trend(gmt_JJA)
mean_trend_gmt_JJA,_,_, mean_p_value_gmt_JJA,_ = linregress(np.arange(44),gmt_mme_JJA)

globalmeantemp=[trends_gmt*10, trends_gmt_DJF*10,trends_gmt_JJA*10]
upwarjet=[trends_ssp245N*10, trends_ssp245N_DJF*10,trends_ssp245N_JJA*10]

######################################################################################################
'''
################################annual#########################################
T_hist585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_1980-2010_zonmean_historical_289x145.nc').ta[
            :, 0, :, :, 0]
T_future585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_2060-2090_zonmean_future_289x145.nc').ta[
              :, 0, :, :, 0]
ua_hist585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_1980-2010_zonmean_historical_289x145.nc').ua[
             :, 0, :, :, 0]
ua_future585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_2060-2090_zonmean_future_289x145.nc').ua[
               :, 0, :, :, 0]
delta_T585 = T_future585 - T_hist585
# Constants
Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization
p = T_hist585.plev  # Pressure levels


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


level = T_hist585.plev / 100
level_label = list(map(int, level.data.tolist()))
beta245 = np.zeros(ua_hist585.shape)
u_prime245 = np.zeros(ua_hist585.shape)
beta585 = np.zeros(ua_hist585.shape)
u_prime585 = np.zeros(ua_hist585.shape)
# Main script execution
for models in range(T_hist585.shape[0]):
    # T = np.mean(T_historical,0)
    # delta_T = np.mean(delta_T,0)
    beta585[models] = calculate_beta(T_hist585[models], delta_T585[models], p)
    u_prime585[models] = calculate_transformed_u(ua_hist585[models], beta585[models], p)
delta_VST_u585 = u_prime585 - ua_hist585
delta_u585 = ua_future585 - ua_hist585
delta_VST_u585 = delta_VST_u585.expand_dims(dim='new_dim', axis=3)
delta_u585 = delta_u585.expand_dims(dim='new_dim', axis=3)

delta_VST_u_100_200_585_annual = [
    (areamean.mask_am(delta_VST_u585[:, 11, 88:104, :]) - areamean.mask_am(delta_VST_u585[:, 9, 88:104, :])),
    (areamean.mask_am(delta_VST_u585[:, 11, 40:56, :]) - areamean.mask_am(delta_VST_u585[:, 9, 40:56, :]))]
delta_u_100_200_585_annual = [(areamean.mask_am(delta_u585[:, 11, 88:104, :]) - areamean.mask_am(delta_u585[:, 9, 88:104, :])),
                       (areamean.mask_am(delta_u585[:, 11, 40:56, :]) - areamean.mask_am(delta_u585[:, 9, 40:56, :]))]
delta_VST_u_100_200_mme585_annual = [np.mean(delta_VST_u_100_200_585_annual[0], 0), np.mean(delta_VST_u_100_200_585_annual[1], 0)]
delta_u_100_200_mme585_annual = [np.mean(delta_u_100_200_585_annual[0], 0), np.mean(delta_u_100_200_585_annual[1], 0)]

#####################################################################
T_hist245 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_DJF_zonmean_historical.nc').ta[
              :, 0, :, :, 0]
T_future245 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_DJF_zonmean_future.nc').ta[
              :, 0, :, :, 0]
ua_hist245 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_DJF_zonmean_historical.nc').ua[
             :, 0, :, :, 0]
ua_future245 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_DJF_zonmean_future.nc').ua[
               :, 0, :, :, 0]
T_hist585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_JJA_zonmean_historical.nc').ta[
            :, 0, :, :, 0]
T_future585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ta_h+ssp585_all_models_JJA_zonmean_future.nc').ta[
              :, 0, :, :, 0]
ua_hist585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_JJA_zonmean_historical.nc').ua[
             :, 0, :, :, 0]
ua_future585 = xr.open_dataset(
    r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/h+ssp585/ua_h+ssp585_all_models_JJA_zonmean_future.nc').ua[
               :, 0, :, :, 0]

delta_T245 = T_future245 - T_hist245
delta_T585 = T_future585 - T_hist585
# Constants
Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization
p = T_hist585.plev  # Pressure levels


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


level = T_hist585.plev / 100
level_label = list(map(int, level.data.tolist()))
beta245 = np.zeros(ua_hist585.shape)
u_prime245 = np.zeros(ua_hist585.shape)
beta585 = np.zeros(ua_hist585.shape)
u_prime585 = np.zeros(ua_hist585.shape)
# Main script execution
for models in range(T_hist585.shape[0]):
    # T = np.mean(T_historical,0)
    # delta_T = np.mean(delta_T,0)
    beta245[models] = calculate_beta(T_hist245[models], delta_T245[models], p)
    u_prime245[models] = calculate_transformed_u(ua_hist245[models], beta245[models], p)
    beta585[models] = calculate_beta(T_hist585[models], delta_T585[models], p)
    u_prime585[models] = calculate_transformed_u(ua_hist585[models], beta585[models], p)

delta_VST_u245 = u_prime245 - ua_hist245
delta_u245 = ua_future245 - ua_hist245
delta_VST_u245 = delta_VST_u245.expand_dims(dim='new_dim', axis=3)
delta_u245 = delta_u245.expand_dims(dim='new_dim', axis=3)

delta_VST_u585 = u_prime585 - ua_hist585
delta_u585 = ua_future585 - ua_hist585
delta_VST_u585 = delta_VST_u585.expand_dims(dim='new_dim', axis=3)
delta_u585 = delta_u585.expand_dims(dim='new_dim', axis=3)

delta_VST_u_100_200_245 = [
    (areamean.mask_am(delta_VST_u245[:, 11, 88:104, :]) - areamean.mask_am(delta_VST_u245[:, 9, 88:104, :])),
    (areamean.mask_am(delta_VST_u245[:, 11, 40:56, :]) - areamean.mask_am(delta_VST_u245[:, 9, 40:56, :]))]
delta_u_100_200_245 = [areamean.mask_am(delta_u245[:, 11, 88:104, :]) - areamean.mask_am(delta_u245[:, 9, 88:104, :]),
                       (areamean.mask_am(delta_u245[:, 11, 40:56, :]) - areamean.mask_am(delta_u245[:, 9, 40:56, :]))]
delta_VST_u_100_200_mme245 = [np.mean(delta_VST_u_100_200_245[0], 0), np.mean(delta_VST_u_100_200_245[1], 0)]
delta_u_100_200_mme245 = [np.mean(delta_u_100_200_245[0], 0), np.mean(delta_u_100_200_245[1], 0)]

delta_VST_u_100_200_585 = [
    (areamean.mask_am(delta_VST_u585[:, 11, 88:104, :]) - areamean.mask_am(delta_VST_u585[:, 9, 88:104, :])),
    (areamean.mask_am(delta_VST_u585[:, 11, 40:56, :]) - areamean.mask_am(delta_VST_u585[:, 9, 40:56, :]))]
delta_u_100_200_585 = [(areamean.mask_am(delta_u585[:, 11, 88:104, :]) - areamean.mask_am(delta_u585[:, 9, 88:104, :])),
                       (areamean.mask_am(delta_u585[:, 11, 40:56, :]) - areamean.mask_am(delta_u585[:, 9, 40:56, :]))]
delta_VST_u_100_200_mme585 = [np.mean(delta_VST_u_100_200_585[0], 0), np.mean(delta_VST_u_100_200_585[1], 0)]
delta_u_100_200_mme585 = [np.mean(delta_u_100_200_585[0], 0), np.mean(delta_u_100_200_585[1], 0)]

delta_u_100_200 = [delta_u_100_200_585_annual,delta_u_100_200_245, delta_u_100_200_585]
delta_VST_u_100_200 = [delta_VST_u_100_200_585_annual,delta_VST_u_100_200_245, delta_VST_u_100_200_585]
delta_u_100_200_mme = [delta_u_100_200_mme585_annual,delta_u_100_200_mme245, delta_u_100_200_mme585]
delta_VST_u_100_200_mme = [delta_VST_u_100_200_mme585_annual,delta_VST_u_100_200_mme245, delta_VST_u_100_200_mme585]
'''


upperlevel=11
lowerlevel=9
#3. hist+ssp585 提取的VST及其std
Rv = 461.5  # Gas constant for water vapor
Lv = 2.5e6  # Latent heat of vaporization

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
        beta[:, lat] = optimize.newton(func, beta_init, args=(delta_T[lat], dT_dp[5, lat], T[5, lat], 5))
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

filepath_T=[f'/home/dongyl/Databank/h+ssp585/zonmean/ta_h+ssp585_all_models_1958-2022_zonmean.nc',
            f'/home/dongyl/Databank/h+ssp585/zonmean/ta_h+ssp585_all_models_1958-2022_zonmean_DJF.nc',
            f'/home/dongyl/Databank/h+ssp585/zonmean/ta_h+ssp585_all_models_1958-2022_zonmean_JJA.nc']
filepath_u=[f'/home/dongyl/Databank/h+ssp585/zonmean/ua_h+ssp585_all_models_1958-2022_zonmean.nc',
            f'/home/dongyl/Databank/h+ssp585/zonmean/ua_h+ssp585_all_models_1958-2022_zonmean_DJF.nc',
            f'/home/dongyl/Databank/h+ssp585/zonmean/ua_h+ssp585_all_models_1958-2022_zonmean_JJA.nc']

for i in range(len(filepath_T)):
    if i == 1:
        T_245 = xr.open_dataset(filepath_T[i]).ta[:,22:63, :, :, 0]
        ua_245 = xr.open_dataset(filepath_u[i]).ua[:, 22:63, :, :, 0]
    else:
        T_245 = xr.open_dataset(filepath_T[i]).ta[:,22:63, :, :, 0]
        ua_245 = xr.open_dataset(filepath_u[i]).ua[:,22:63, :, :, 0]
    years_hist = np.arange(1980, 2021)
    trends_hist=np.zeros((T_245.shape[0],T_245.shape[3]))
    p_values_hist=np.zeros((T_245.shape[0],T_245.shape[3]))
    delta_VST_u_hist_100_200=np.zeros(T_245.shape[0])
    beta_hist = np.zeros((T_245.shape[0], 19, 145))
    u_prime_hist = np.zeros((T_245.shape[0], 19, 145))
    p = T_245.plev
    for model in range(T_245.shape[0]):
        for j in range(T_245.shape[3]):
            slope, _, _, p_value, _ = stats.linregress(years_hist, T_245[model,:,5,j])
            trends_hist[model,j] = slope
            p_values_hist[model,j] = p_value
    delta_T_hist = trends_hist*80
    # Main script execution
    for models in range(T_245.shape[0]):
        # T = np.mean(T_historical,0)
        # delta_T = np.mean(delta_T,0)
        beta_hist[models] = calculate_beta(np.mean(T_245[models],0), delta_T_hist[models],p)
        u_prime_hist[models] = calculate_transformed_u(np.mean(ua_245[models],0), beta_hist[models],p)
    delta_VST_u245 = u_prime_hist - ua_245.mean(axis=1) #保证delta_VST_u245为Dataarray,便于后续使用areamean.mask_am函数
    delta_VST_u245 = delta_VST_u245.expand_dims(dim='new_dim', axis=3)/80 ####恢复1年的向上抬升贡献
    #delta_VST_u245 = np.expand_dims(delta_VST_u245, 3)
    delta_VST_u_100_200_245_N =(areamean.mask_am(delta_VST_u245[:, upperlevel, 88:104, :]) - areamean.mask_am(delta_VST_u245[:, lowerlevel, 88:104, :]))
    delta_VST_u_100_200_mme245_N = np.mean(delta_VST_u_100_200_245_N, 0)
    if i==0:
        annual_VST_trend_100minus200=delta_VST_u_100_200_245_N
    if i == 1:
        DJF_VST_trend_100minus200=delta_VST_u_100_200_245_N
    if i==2:
        JJA_VST_trend_100minus200=delta_VST_u_100_200_245_N

hist_all_forcing_DJF = xr.open_dataset(r'/home/dongyl/Databank/h+ssp585/zonmean/ua_h+ssp585_all_models_1958-2022_zonmean_DJF.nc').ua[:,22:63]
hist_all_forcing_JJA = xr.open_dataset(r'/home/dongyl/Databank/h+ssp585/zonmean/ua_h+ssp585_all_models_1958-2022_zonmean_JJA.nc').ua[:,22:63]
hist_all_forcing_annual = xr.open_dataset(r'/home/dongyl/Databank/h+ssp585/zonmean/ua_h+ssp585_all_models_1958-2022_zonmean.nc').ua[:,22:63]
hist_GHG_DJF = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/hist-GHG/ua_hist-GHG_all_models_1958-2019_288x145_zonmeanDJF.nc').ua[:,22:]


DJF_100minus200 = hist_all_forcing_DJF[:,:,upperlevel]-hist_all_forcing_DJF[:,:,lowerlevel]
JJA_100minus200 = hist_all_forcing_JJA[:,:,upperlevel]-hist_all_forcing_JJA[:,:,lowerlevel]
annual_100minus200 = hist_all_forcing_annual[:,:,upperlevel]-hist_all_forcing_annual[:,:,lowerlevel]

DJF_mean_100minus200=np.nanmean(DJF_100minus200,axis=0)
JJA_mean_100minus200=np.nanmean(JJA_100minus200,axis=0)
annual_mean_100minus200=np.nanmean(annual_100minus200,axis=0)

def areamean_func_4D(ndarray,slice,lat_ds):
    months_label, time_label, lat_label,lon_label = ndarray.shape
    nrows = lat_label
    latsr = np.deg2rad(lat_ds['lat'].values[slice]).reshape((nrows, 1))
    weight_matrix = np.repeat(np.cos(latsr), lon_label, axis=1)
    ds_w = ndarray * weight_matrix
    weight_sum = copy.deepcopy(weight_matrix)
    #weight_sum[ds_w[0, 0, 0].isnull().data] = np.nan
    obs_mean_areamean = ds_w.sum(dim=['lat','lon']) / np.nansum(weight_sum)
    return obs_mean_areamean
def areamean_func_3D(ndarray,slice,lat_ds):
    time_label, lat_label,lon_label = ndarray.shape
    nrows = lat_label
    latsr = np.deg2rad(lat_ds['lat'].values[slice]).reshape((nrows, 1))
    #weight_matrix = np.repeat(np.cos(latsr), 1, axis=1)
    ds_w = ndarray * np.cos(latsr)
    weight_sum = copy.deepcopy(np.cos(latsr))
    #weight_sum[ds_w[0, 0, 0].isnull().data] = np.nan
    obs_mean_areamean = ds_w.sum(1) / np.nansum(weight_sum)
    return obs_mean_areamean
###########集合平均结果###########
DJF_areamean_100minus200_N = areamean_func_4D(DJF_100minus200[:,:,slice(89,105),:],slice(89,105),hist_GHG_DJF)
JJA_areamean_100minus200_N = areamean_func_4D(JJA_100minus200[:, :, slice(89, 105), :], slice(89, 105),hist_GHG_DJF)
annual_areamean_100minus200_N = areamean_func_4D(annual_100minus200[:, :, slice(89, 105), :],slice(89, 105), hist_GHG_DJF)

DJF_hist_trend_100minus200, _ = dyl.calculate_trend_2D_sc(DJF_areamean_100minus200_N)
JJA_hist_trend_100minus200, _ = dyl.calculate_trend_2D_sc(JJA_areamean_100minus200_N)
annual_hist_trend_100minus200, _ = dyl.calculate_trend_2D_sc(annual_areamean_100minus200_N)

delta_u_100_200=[annual_hist_trend_100minus200*10,DJF_hist_trend_100minus200*10,JJA_hist_trend_100minus200*10]
delta_VST_u_100_200=[annual_VST_trend_100minus200*10,DJF_VST_trend_100minus200*10,JJA_VST_trend_100minus200*10]


np.random.seed(0)
# 使用 15 个不同的符号表示不同的模式
markers = ['o', 'v', '^', '<', '>', '8', 's', 'p', '1', 'h', 'H', 'D', 'd', 'P', 'X']

title = ['annual','DJF','JJA']
fig = plt.figure(figsize=(6, 6),dpi=600)
axes = [[.12, .7, .28, .24], [.5, .7, .28, .24],
       [.12, .39, .28, .24], [.5, .39, .28, .24],
        [.12, .08, .28, .24],  [.5, .08, .28, .24]]
models = ['ACCESS-CM2', 'ACCESS-ESM1-5', 'AWI-CM-1-1-MR', 'BCC-CSM2-MR', 'CanESM5', 'CAS-ESM2-0', 'CESM2-WACCM',
          'CMCC-CM2-SR5',
          'CMCC-ESM2', 'EC-Earth3', 'EC-Earth3-Veg', 'EC-Earth3-Veg-LR', 'FGOALS-f3-L', 'FGOALS-g3', 'FIO-ESM-2-0',
          'GFDL-ESM4', 'IITM-ESM', 'INM-CM5-0', 'IPSL-CM6A-LR', 'KACE-1-0-G', 'KIOST-ESM', 'MIROC6', 'MPI-ESM1-2-HR',
          'MRI-ESM2-0', 'NESM3', 'NorESM2-LM', 'NorESM2-MM', 'TaiESM1', 'MME']

sequence = ['a', 'b', 'c', 'd','e','f']
for j in range(6):
    if j%2 == 1:
        ax = fig.add_axes(axes[j])
        i=j//2
        # 线性回归分析
        slope, intercept, r_value, p_value, std_err = stats.linregress(delta_u_100_200[i],
                                                                       delta_VST_u_100_200[i])
        # 绘制回归直线
        line = slope * delta_u_100_200[i] + intercept
        ax.plot(delta_u_100_200[i], line, color='red', linewidth=.5)

        # 绘制散点图
        for m in range(28):
            ax.scatter(delta_u_100_200[i][m], delta_VST_u_100_200[i][m], s=10,
                       marker=markers[m % 15], label=models[m])
        ax.scatter(np.nanmean(delta_u_100_200[i]), np.nanmean(delta_VST_u_100_200[i]), marker='*', s=50, color='k',
                   zorder=20,
                   label=models[-1])
        # 添加图例、标签和标题
        if i<2:
            ax.set_ylabel('Δu$_{100}$-Δu$_{200}$ (VST)', fontsize=7)
        if i==2:
            ax.set_ylabel('Δu$_{100}$-Δu$_{200}$ (VST)', fontsize=7)
            ax.set_xlabel('Δu$_{100}$-Δu$_{200}$', fontsize=7) #(m s$^{-1}$ decade$^{-1}$)
        # 在图上标注斜率和显著性
        ax.annotate(f'Slope: {slope:.2f}\nR = ${r_value:.2f}$\np < 0.01', xy=(0.6, 0.25), xycoords='axes fraction',
                    horizontalalignment='left', verticalalignment='top', fontsize=7)
        # min_val = min(delta_u_100_200[i].min(), delta_VST_u_100_200[i].min()) - 0.1
        # max_val = max(delta_u_100_200[i].max(), delta_VST_u_100_200[i].max()) + 0.1
        # ax.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', linewidth=1)
        ax.set_xlim(-.1, .7)
        ax.set_ylim(-.1, .7)
        ax.plot([-.1, .7], [-.1, .7], color='grey', linestyle='--', linewidth=.5)
        # 增大坐标轴标签和刻度的字体大小
        ax.tick_params(axis='both', which='major', labelsize=7,direction='in')
        # ax.set_title('Based on T', loc='left', fontsize=14)
        ax.set_title(title[i], loc='right', fontsize=7)
        ax.text(-0.15, 1.15, sequence[j], transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='left')

    else:
        ax = fig.add_axes(axes[j])
        i = j // 2
        # 线性回归分析
        slope, intercept, r_value, p_value, std_err = stats.linregress(globalmeantemp[i], upwarjet[i])
        # 扩展自变量范围
        extended_gmt = np.linspace(min(globalmeantemp[i]) - .05, max(globalmeantemp[i]) + .05, 100)
        # 计算扩展范围内的回归值
        extended_line = slope * extended_gmt + intercept
        ax.plot(extended_gmt, extended_line, color='red', linewidth=.8)
        # 绘制散点图
        for m in range(h_ssp245_100minus200N.shape[0]):
            ax.scatter(globalmeantemp[i][m], upwarjet[i][m], s=5,
                       marker=markers[m % 15], label=models[m])
        ax.scatter(np.nanmean(globalmeantemp[i]), np.nanmean(upwarjet[i]), marker='*', s=30, color='k',
                   zorder=20,
                   label='mme')
        # 添加图例、标签和标题
        ax.set_ylabel('Δu$_{100}$-Δu$_{200}$', fontsize=7)# (m s$^{-1}$ decade$^{-1}$)
        if i == 2:
            ax.set_xlabel('global mean temperature', fontsize=7)# (K decade$^{-1}$)
            # 在图上标注斜率和显著性
            ax.annotate(f'Slope: {slope:.2f}\nR = ${r_value:.2f}$\np < 0.05', xy=(0.6, 0.25), xycoords='axes fraction',
                        horizontalalignment='left', verticalalignment='top', fontsize=6)
        else:
            ax.annotate(f'Slope: {slope:.2f}\nR = ${r_value:.2f}$\np < 0.01', xy=(0.6, 0.25), xycoords='axes fraction',
                        horizontalalignment='left', verticalalignment='top', fontsize=6)
        ax.set_xlim(0, .6)
        ax.set_ylim(0, .6)
        ax.plot([-.1, .6], [-.1, .6], color='grey', linestyle='--', linewidth=.5)
        # 增大坐标轴标签和刻度的字体大小
        ax.tick_params(axis='both', which='major', labelsize=7, direction='in')
        # ax.set_title('Based on T', loc='left', fontsize=14)
        ax.set_title(title[i], loc='right', fontsize=7)
        ax.text(-0.15, 1.15, sequence[j], transform=ax.transAxes, fontsize=14, fontweight='bold', va='top', ha='left')
        # 增大坐标轴标签和刻度的字体大小
plt.savefig('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/figures/u100-u200_VST_ssp_and_gmt_scatter.png',dpi=600)
plt.legend(fontsize=5, frameon=False,bbox_to_anchor=(1.1, 1.6),loc='center left')
plt.show()


