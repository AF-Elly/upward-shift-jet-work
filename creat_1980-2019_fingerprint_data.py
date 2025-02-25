############生成1980-2019噪音和信号序列###########
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy import signal
from cartopy.util import add_cyclic_point
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import areamean_dhq as dhq
import copy
import dyl_function_slope as dyl
import cmaps
import os

######################以下，计算picontrol噪音#####################
directory = '/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/picontrol/'
files = os.listdir(directory)
picontrol_name = ['ACCESS-CM2', 'BCC-CSM2-MR', 'CanESM5','EC‐Earth3','FGOALS-g3','IITM-ESM','INM-CM4-8','INM-CM5-0',
    'IPSL-CM6A-LR','KACE-1-0-G','MIROC6','MPI-ESM1-2-HR','MPI-ESM1-2-LR','MRI-ESM2-0','NorESM2-LM','NorESM2-MM','TaiESM1']

for i,file in enumerate(files):
    if file.endswith('.nc'):
        print(file)
        file_path = os.path.join(directory, file)
        picontrol = xr.open_dataset(file_path).ua[:,0]
        noise_amount = picontrol.shape[0]//8
        years = noise_amount*8
        print(noise_amount)
        picontrol_S = dhq.mask_am(picontrol[:years,40:56,:])
        picontrol_N = dhq.mask_am(picontrol[:years,89:105,:])
        picontrol_S_detrend = signal.detrend(picontrol_S, axis=0, type='linear',
                                            bp=0, overwrite_data=False)
        picontrol_N_detrend = signal.detrend(picontrol_N, axis=0, type='linear',
                                            bp=0, overwrite_data=False)
        #ax.plot(picontrol_S_detrend, c="b")
        #plt.show()
        #noise_S = picontrol_S.to_numpy()
        #noise_N = picontrol_N.to_numpy()
        noise_S = np.reshape(picontrol_S_detrend,(noise_amount,8))
        noise_N = np.reshape(picontrol_N_detrend,(noise_amount, 8))
        noise_S_anom = noise_S - np.expand_dims(noise_S.mean(axis=1),1)
        noise_N_anom = noise_N - np.expand_dims(noise_N.mean(axis=1),1)
        #ax.plot(noise_S_anom, c="r")
        if i==0:
            noise_S_anom_old = np.zeros((1,8))
            noise_N_anom_old = np.zeros((1,8))
        noise_S_anom_old = np.concatenate((noise_S_anom_old, noise_S_anom), axis=0)
        noise_N_anom_old = np.concatenate((noise_N_anom_old, noise_N_anom), axis=0)

noise_S_anom = noise_S_anom_old[1:]
noise_N_anom = noise_N_anom_old[1:]
print(len(noise_S_anom))

######################以下，计算GHG aer nat噪音#####################
directory = '/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/GHG_aer_nat_noise/5yrmean_u100minusu200/'
files = os.listdir(directory)
model_name = ["ACCESS-CM2","ACCESS-ESM1-5","BCC-CSM2-MR","CanESM5", "FGOALS-g3",
              "GFDL-ESM4","IPSL-CM6A-LR","MIROC6","MRI-ESM2-0","NorESM2-LM"]
fig2 = plt.figure()
ax = fig2.add_axes([0.1,0.1,0.8,0.8])
for i,file in enumerate(files):
    if file.endswith('.nc'):
        print(file)
        file_path = os.path.join(directory, file)
        #picontrol_100 = xr.open_dataset(file_path).ua[:,11]
        #picontrol_200 = xr.open_dataset(file_path).ua[:, 9]
        picontrol = xr.open_dataset(file_path).ua[:,0]
        noise_amount = picontrol.shape[0]//8
        years = noise_amount*8
        print(noise_amount)
        lat = xr.open_dataset(file_path).lat.values
        lat_indices_S = np.where((lat >= -40) & (lat <= -20))[0]
        lat_indices_N = np.where((lat >= 20) & (lat <= 40))[0]
        picontrol_S = dhq.mask_am(picontrol[:years,lat_indices_S,:])
        picontrol_N = dhq.mask_am(picontrol[:years,lat_indices_N,:])
        #picontrol_S_detrend = signal.detrend(picontrol_S, axis=0, type='linear',
        #                                    bp=0, overwrite_data=False)
        #picontrol_N_detrend = signal.detrend(picontrol_N, axis=0, type='linear',
        #                                   bp=0, overwrite_data=False)
        #plt.plot(picontrol_S_detrend, c="b")
        picontrol_N = np.expand_dims(picontrol_N.to_numpy(), axis=1)
        picontrol_S = np.expand_dims(picontrol_S.to_numpy(), axis=1)
        #noise_S = np.expand_dims(picontrol_S, axis=1)
        #noise_N = np.expand_dims(picontrol_N, axis=1)
        noise_S = np.reshape(picontrol_S,(noise_amount,8))
        noise_N = np.reshape(picontrol_N,(noise_amount, 8))
        noise_S_anom = noise_S - np.expand_dims(noise_S.mean(axis=1),1)
        noise_N_anom = noise_N - np.expand_dims(noise_N.mean(axis=1),1)
        ax.plot(noise_N_anom[0], c="k")
        #if i==0:
        #    noise_S_anom_old = np.zeros((1,12))
        #    noise_N_anom_old = np.zeros((1,12))
        noise_S_anom_old = np.concatenate((noise_S_anom_old, noise_S_anom), axis=0)
        noise_N_anom_old = np.concatenate((noise_N_anom_old, noise_N_anom), axis=0)
noise_S_anom = noise_S_anom_old[1:]
noise_N_anom = noise_N_anom_old[1:]
plt.show()
# 设置随机种子（可选）以便结果可重复
np.random.seed(0)
# 对 noise_S_anom 生成独立的随机索引
indices_S = np.random.permutation(noise_S_anom.shape[0])
num_pieces = noise_S_anom.shape[0]//2
group1_indices_S = indices_S[:num_pieces]
group2_indices_S = indices_S[num_pieces:]
# 对 noise_N_anom 生成独立的随机索引
indices_N = np.random.permutation(noise_N_anom.shape[0])
group1_indices_N = indices_N[:num_pieces]
group2_indices_N = indices_N[num_pieces:]
# 根据各自的随机索引分割 noise_S_anom 和 noise_N_anom
noise_S_anom1 = noise_S_anom[group1_indices_S]
noise_S_anom2 = noise_S_anom[group2_indices_S]
noise_N_anom1 = noise_N_anom[group1_indices_N]
noise_N_anom2 = noise_N_anom[group2_indices_N]
'''
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/noise1_1980-2019_20_40_N_5yrmean.dat', noise_N_anom1)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/noise2_1980-2019_20_40_N_5yrmean.dat', noise_N_anom2)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/noise1_1980-2019_20_40_S_5yrmean.dat', noise_S_anom1)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/noise2_1980-2019_20_40_S_5yrmean.dat', noise_S_anom2)
'''
'''
ERA5 = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/ERA5/u_ERA5_1980-2019_5yrmean_u100minusu200_anom.nc').u100minusu200_anom_5yrmean[:,0]
MERRA2 = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/MERRA-2/u_MERRA2_1980-2019_5yrmean_u100minusu200_anom.nc').u100minusu200_anom_5yrmean[:,0]
JRA55 = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/JRA-55/u_jra55_1980-2019_5yrmean_u100minusu200_anom.nc').u100minusu200_anom_5yrmean[:,0]
h_ssp585= xr.open_dataset(r'/home/dongyl/Databank/h+ssp585/zonmean/ua_h+ssp585_all_models_1958-2022_zonmean.nc').ua[:,22:62,:,:,:]

obs_data = [ERA5,JRA55,MERRA2]
obs = np.nanmean(obs_data,axis=0)
h_ssp585_mme = h_ssp585.mean(axis=0)
#h_ssp585_mme = xr.DataArray(h_ssp585_mme, dims=['year','level', 'lat', 'lon'], coords={'year': np.arange(1, 41),'level':h_ssp585.plev, 'lat': h_ssp585.lat, 'lon': h_ssp585.lon})

# 定义一个函数来计算5年非重叠平均
def calculate_5year_rolling_avg(data, window=5):
    # 获取年份数据
    #years = data.year.values
    # 切分数据为每5年的区间
    grouped = [data.isel(time=slice(i, i + 5)) for i in range(0, 40, 5)]
    # 计算每5年的均值
    means = [group.mean(dim='time') for group in grouped]
    # 将结果组合为一个新的DataArray
    return xr.concat(means, dim='time')

# 对ERA5和JRA55数据进行5年非重叠平均
obs_avg = np.nanmean(obs_data,axis=0)
h_ssp585_mme_avg = calculate_5year_rolling_avg(h_ssp585_mme)

ERA5_S = dhq.mask_am(ERA5[:,40:56,:])
ERA5_N = dhq.mask_am(ERA5[:,89:105,:])
MERRA2_S = dhq.mask_am(MERRA2[:,40:56,:])
MERRA2_N = dhq.mask_am(MERRA2[:,89:105,:])
JRA55_N = dhq.mask_am(JRA55[:,40:56,:])
JRA55_S = dhq.mask_am(JRA55[:,89:105,:])

sigmme_20_40_S = dhq.mask_am(h_ssp585_mme_avg[:,11,40:56,:]-h_ssp585_mme_avg[:,9,40:56,:])
sigmme_20_40_N= dhq.mask_am(h_ssp585_mme_avg[:,11,89:105,:]-h_ssp585_mme_avg[:,9,89:105,:])

obsmme_20_40_N_anom = np.nanmean([ERA5_N,JRA55_N,MERRA2_N],axis=0)
obsmme_20_40_S_anom = np.nanmean([ERA5_S,JRA55_S,MERRA2_S],axis=0)
sigmme_20_40_S_anom = sigmme_20_40_S-np.expand_dims(sigmme_20_40_S.mean(axis=0),0)
sigmme_20_40_N_anom = sigmme_20_40_N-np.expand_dims(sigmme_20_40_N.mean(axis=0),0)

obs_sig_N = [obsmme_20_40_N_anom, sigmme_20_40_N_anom]
obs_sig_S = [obsmme_20_40_S_anom, sigmme_20_40_S_anom]
#data = {'obs': [1, 2, 3, 4], 'Column2': [5, 6, 7, 8]}
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obssigall_20_40_N_1980-2019_5yrmean_anom.dat', obs_sig_N)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obssigall_20_40_S_1980-2019_5yrmean_anom.dat', obs_sig_S)

##############################################################################
hist_GHG = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/hist-GHG/ua_hist-GHG_all_models_1958-2019_288x145_zonmean.nc').ua[:,22:62,:,:,:]
hist_aer = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/hist-aer/ua_hist-aer_all_models_1958-2019_288x145_zonmean.nc').ua[:,22:62,:,:,:]
hist_nat = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/hist-nat/ua_hist-nat_all_models_1958-2019_288x145_zonmean.nc').ua[:,22:62,:,:,:]

hist_GHG_mme = hist_GHG.mean(axis=0)
#hist_GHG_mme = xr.DataArray(hist_GHG_mme, dims=['year','level', 'lat', 'lon'], coords={'year': np.arange(1, 41),'level':hist_GHG.plev, 'lat': hist_GHG.lat, 'lon': hist_GHG.lon})
hist_aer_mme = hist_aer.mean(axis=0)
#hist_aer_mme = xr.DataArray(hist_aer_mme, dims=['year','level', 'lat', 'lon'], coords={'year': np.arange(1, 41),'level':hist_GHG.plev, 'lat': hist_GHG.lat, 'lon': hist_GHG.lon})
hist_nat_mme = hist_nat.mean(axis=0)
#hist_nat_mme = xr.DataArray(hist_nat_mme, dims=['year','level', 'lat', 'lon'], coords={'year': np.arange(1, 41),'level':hist_GHG.plev, 'lat': hist_GHG.lat, 'lon': hist_GHG.lon})

# 对ERA5和JRA55数据进行5年非重叠平均
hist_GHG_avg = calculate_5year_rolling_avg(hist_GHG_mme)
hist_aer_avg = calculate_5year_rolling_avg(hist_aer_mme)
hist_nat_avg = calculate_5year_rolling_avg(hist_nat_mme)

GHG_20_40_S = dhq.mask_am(hist_GHG_avg[:,11,40:56,:]-hist_GHG_avg[:,9,40:56,:])
GHG_20_40_N = dhq.mask_am(hist_GHG_avg[:,11,89:105,:]-hist_GHG_avg[:,9,89:105,:])
aer_20_40_S = dhq.mask_am(hist_aer_avg[:,11,40:56,:]-hist_aer_avg[:,9,40:56,:])
aer_20_40_N= dhq.mask_am(hist_aer_avg[:,11,89:105,:]-hist_aer_avg[:,9,89:105,:])
nat_20_40_S = dhq.mask_am(hist_nat_avg[:,11,40:56,:]-hist_nat_avg[:,9,40:56,:])
nat_20_40_N= dhq.mask_am(hist_nat_avg[:,11,89:105,:]-hist_nat_avg[:,9,89:105,:])

GHG_20_40_S_anom = GHG_20_40_S-np.expand_dims(GHG_20_40_S.mean(axis=0),0)
GHG_20_40_N_anom = GHG_20_40_N-np.expand_dims(GHG_20_40_N.mean(axis=0),0)
aer_20_40_S_anom = aer_20_40_S-np.expand_dims(aer_20_40_S.mean(axis=0),0)
aer_20_40_N_anom = aer_20_40_N-np.expand_dims(aer_20_40_N.mean(axis=0),0)
nat_20_40_S_anom = nat_20_40_S-np.expand_dims(nat_20_40_S.mean(axis=0),0)
nat_20_40_N_anom = nat_20_40_N-np.expand_dims(nat_20_40_N.mean(axis=0),0)

GHG_aer_N = [obsmme_20_40_N_anom, GHG_20_40_N_anom, aer_20_40_N_anom, nat_20_40_N_anom]
GHG_aer_S = [obsmme_20_40_S_anom, GHG_20_40_S_anom, aer_20_40_S_anom, nat_20_40_S_anom]
sig_GHG_N = [obsmme_20_40_N_anom, GHG_20_40_N_anom]
sig_GHG_S = [obsmme_20_40_S_anom, GHG_20_40_S_anom]
sig_aer_N = [obsmme_20_40_N_anom, aer_20_40_N_anom]
sig_aer_S = [obsmme_20_40_S_anom, aer_20_40_S_anom]
sig_nat_N = [obsmme_20_40_N_anom, nat_20_40_N_anom]
sig_nat_S = [obsmme_20_40_S_anom, nat_20_40_S_anom]

np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obs_GHG_aer_nat_ens3run_20_40_N_1980-2019_5yrmean_anom.dat', GHG_aer_N)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obs_GHG_aer_nat_ens3run_20_40_S_1980-2019_5yrmean_anom.dat', GHG_aer_S)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obs_GHG_ens3run_20_40_N_1980-2019_5yrmean_anom.dat', sig_GHG_N)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obs_GHG_ens3run_20_40_S_1980-2019_5yrmean_anom.dat', sig_GHG_S)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obs_aer_ens3run_20_40_N_1980-2019_5yrmean_anom.dat', sig_aer_N)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obs_aer_ens3run_20_40_S_1980-2019_5yrmean_anom.dat', sig_aer_S)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obs_nat_ens3run_20_40_N_1980-2019_5yrmean_anom.dat', sig_nat_N)
np.savetxt('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/obs_nat_ens3run_20_40_S_1980-2019_5yrmean_anom.dat', sig_nat_S)
'''