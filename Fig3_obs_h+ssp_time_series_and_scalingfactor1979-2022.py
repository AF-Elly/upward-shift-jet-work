import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from cartopy.util import add_cyclic_point
from scipy.stats import linregress
import areamean_dhq as areamean
import dyl_function_slope as dyl
plt.rcParams['font.family'] = 'Arial'

ERA5N = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/ERA5/u_ERA5_197901-202212_areamean_N_100_200hPa.nc').u[:,:,0,0]
MERRA2N = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/MERRA-2/u_MERRA-2_198001-202212_areamean_N_100_200hPa.nc').U[:,:,0,0]
JRA55N = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/JRA-55/u_JRA-55_197901-202212_areamean_N_100_200hPa.nc').UGRD_GDS0_ISBL_S123[:,:,0,0]
amipN = xr.open_dataset(r'/home/dongyl/Databank/amip/ua/ua_amip_all_models9_197901-201412_yearmean_areamean_100_200hPa_N.nc').ua[:,:,:,0,0]

ERA5_100minus200N = ERA5N[:,0]-ERA5N[:,1]
MERRA2_100minus200N = MERRA2N[:,1]-MERRA2N[:,0]
JRA55_100minus200N = JRA55N[:,0]-JRA55N[:,1]
amip_100minus200N=amipN[:,:,1]-amipN[:,:,0]

ERA5S = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/ERA5/u_ERA5_197901-202212_areamean_S_100_200hPa.nc').u[:,:,0,0]
MERRA2S = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/MERRA-2/u_MERRA-2_197901-202212_areamean_S_100_200hPa.nc').U[:,:,0,0]
JRA55S = xr.open_dataset(r'/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/obs-data/JRA-55/u_JRA-55_197901-202212_areamean_S_100_200hPa.nc').UGRD_GDS0_ISBL_S123[:,:,0,0]
amipS = xr.open_dataset(r'/home/dongyl/Databank/amip/ua/ua_amip_all_models9_197901-201412_yearmean_areamean_100_200hPa_S.nc').ua[:,:,:,0,0]

ERA5_100minus200S = ERA5S[:,0]-ERA5S[:,1]
MERRA2_100minus200S = MERRA2S[:,1]-MERRA2S[:,0]
JRA55_100minus200S = JRA55S[:,0]-JRA55S[:,1]
amip_100minus200S= amipS[:,:,1]-amipS[:,:,0]

h_ssp585 = xr.open_dataset(r'/home/dongyl/Databank/h+ssp585/zonmean/ua_h+ssp585_all_models_1958-2022_zonmean.nc').ua[:,21:]
h_ssp585_100minus200=h_ssp585[:,:,11]-h_ssp585[:,:,9]
h_ssp585_100minus200_S = areamean.mask_am4D(h_ssp585_100minus200[:,:,slice(40,56)])
h_ssp585_100minus200_N = areamean.mask_am4D(h_ssp585_100minus200[:,:,slice(89,105)])
h_ssp585mme_100minus200=h_ssp585[:,:,11]-h_ssp585[:,:,9]
h_ssp585mme_100minus200= h_ssp585mme_100minus200.mean(axis=0)
h_ssp585mme_100minus200_S = areamean.mask_am(h_ssp585mme_100minus200[:,slice(40,56)])
h_ssp585mme_100minus200_N = areamean.mask_am(h_ssp585mme_100minus200[:,slice(89,105)])
trends_ssp585N, p_values_ssp585N = dyl.calculate_trend(h_ssp585_100minus200_N)
trends_ssp585S, p_values_ssp585S = dyl.calculate_trend(h_ssp585_100minus200_S)
mean_trend_ssp585N,_,_, mean_p_value_ssp585N,_ = linregress(np.arange(44),h_ssp585mme_100minus200_N)
mean_trend_ssp585S,_,_, mean_p_value_ssp585S,_ = linregress(np.arange(44),h_ssp585mme_100minus200_S)


def calculate_trends(data):
    trends = []
    for model_data in data:
        # Assuming time in years for x-axis starting from 1979
        x = np.arange(1979, 1979 + model_data.size)
        slope, intercept, r_value, p_value, std_err = linregress(x, model_data)
        trends.append(slope*10)         ####################注意！！！！这里为了使单位为/decade，对函数内部做了改动：*120###
    return trends

# Calculate the trends for each model and observational data
trends_ERA5N,_,_,p_value_ERA5N,_ = linregress(np.arange(44), ERA5_100minus200N)
trends_MERRA2N,_,_,p_value_MERRA2N,_ = linregress(np.arange(43), MERRA2_100minus200N)
trends_JRA55N,_,_,p_value_JRA55N,_ = linregress(np.arange(44), JRA55_100minus200N)
trends_amipN, p_values_amipN = dyl.calculate_trend(amip_100minus200N)
mean_trend_amipN,_,_, mean_p_value_amipN,_ = linregress(np.arange(36),np.mean(amip_100minus200N,0))

trends_ERA5S,_,_,p_value_ERA5S,_ = linregress(np.arange(44), ERA5_100minus200S)
trends_MERRA2S,_,_,p_value_MERRA2S,_ = linregress(np.arange(43), MERRA2_100minus200S)
trends_JRA55S,_,_,p_value_JRA55S,_ = linregress(np.arange(44), JRA55_100minus200S)
trends_amipS, p_values_amipS = dyl.calculate_trend(amip_100minus200S)
mean_trend_amipS,_,_, mean_p_value_amipS,_ = linregress(np.arange(36),np.mean(amip_100minus200S,0))



fig = plt.figure(figsize=(8, 6),dpi=600)
axes = [[0.1,0.56,0.5,0.34], [0.72,0.56,0.25,0.34],[0.1,0.08,0.5,0.34],[0.72,0.08,0.25,0.34]]
# New subplot for trends
scenarios = ['HIST+SSP585\n(1979-2022)', 'OBS', 'AMIP']
x_pos = np.arange(len(scenarios))
obs_trends = [[trends_ERA5N*10, trends_MERRA2N*10, trends_JRA55N*10], [trends_ERA5S*10, trends_MERRA2S*10, trends_JRA55S*10]]
obs_p_values = [[p_value_ERA5N, p_value_MERRA2N, p_value_JRA55N], [p_value_ERA5S, p_value_MERRA2S, p_value_JRA55S]]
#all_trends = [trends_ssp585*120 + [mean_trend_ssp585*120], obs_trends]
# Plotting individual model and observation trends as scatter points
sequence = ['a', 'b','c', 'd']
obs_color=['green','blue','purple']
obs_label = ['ERA5', 'MERRA-2', 'JRA-55']
title = ['20°-40° N','20°-40° S']
# 计算扣除1979-2008 均值后的u100-u200
# 计算扣除1979-2008 均值后的u100-u200
h_ssp585_100minus200 = [h_ssp585_100minus200_N-h_ssp585_100minus200_N.mean(axis=1).broadcast_like(h_ssp585_100minus200_N),
                        h_ssp585_100minus200_S-h_ssp585_100minus200_S.mean(axis=1).broadcast_like(h_ssp585_100minus200_S)]

ERA5_100minus200 = [ERA5_100minus200N-ERA5_100minus200N[:30].mean(axis=0), ERA5_100minus200S-ERA5_100minus200S[:30].mean(axis=0)]
MERRA2_100minus200 = [MERRA2_100minus200N-MERRA2_100minus200N[:29].mean(axis=0), MERRA2_100minus200S-MERRA2_100minus200S[:29].mean(axis=0)]
JRA55_100minus200 = [JRA55_100minus200N-JRA55_100minus200N[:30].mean(axis=0), JRA55_100minus200S-JRA55_100minus200S[:30].mean(axis=0)]
amip_100minus200 = [amip_100minus200N-amip_100minus200N[:,:30].mean(axis=1).broadcast_like(amip_100minus200N),
                    amip_100minus200S-amip_100minus200S[:,:30].mean(axis=1).broadcast_like(amip_100minus200S)]
p_values_ssp585 = [p_values_ssp585N, p_values_ssp585S]
#p_values_ssp245 = [p_values_ssp245N, p_values_ssp245S]
p_values_amip = [p_values_amipN, p_values_amipS]
trends_ssp585 = [trends_ssp585N, trends_ssp585S]
#trends_ssp245 = [trends_ssp245N, trends_ssp245S]
trends_amip = [trends_amipN, trends_amipS]
mean_trend_ssp585 = [mean_trend_ssp585N*10, mean_trend_ssp585S*10]
#mean_trend_ssp245 = [mean_trend_ssp245N*10, mean_trend_ssp245S*10]
mean_trend_amip = [mean_trend_amipN*10, mean_trend_amipS*10]


# 为每个条形图添加高度标注
def add_bar_labels(color, xloc, yheight, text):
    ax2.text(xloc, yheight, f'{text:.2f}', ha='center', va='bottom', fontsize=6,
             color=color)

# Original plot
# Plotting the model ensemble mean and observational data
for i in range (1):
    ax1 = fig.add_axes(axes[i*2])
    ax2 = fig.add_axes(axes[i*2+1])
    ax1.plot(np.arange(1979, 2023), h_ssp585_100minus200[i][:,:44].mean(axis=0),
             label='HIST+SSP585', color='red', linewidth=2)
    ax1.fill_between(np.arange(1979, 2023), h_ssp585_100minus200[i][:,:44].mean(axis=0) + h_ssp585_100minus200[i][:,:44].std(axis=0),
                        h_ssp585_100minus200[i][:,:44].mean(axis=0) - h_ssp585_100minus200[i][:,:44].std(axis=0),
                        color='red', alpha=0.1)
    #ax1.plot(np.arange(1979, 2023), h_ssp245_100minus200[i][:,:44].mean(axis=0),
    #         label='MME (HIST+SSP245)', color='orange', linewidth=2)
    #ax1.fill_between(np.arange(1979, 2023), h_ssp245_100minus200[i][:,:44].mean(axis=0) + h_ssp245_100minus200[i][:,:44].std(axis=0),
    #                 h_ssp245_100minus200[i][:,:44].mean(axis=0) - h_ssp245_100minus200[i][:,:44].std(axis=0),
    #                 color='orange', alpha=0.1)

    ax1.plot(np.arange(1979, 2023), ERA5_100minus200[i], label='ERA5', color='green', linewidth=2)
    ax1.plot(np.arange(1980, 2023), MERRA2_100minus200[i], label='MERRA-2', color='blue', linewidth=2)
    ax1.plot(np.arange(1979, 2023), JRA55_100minus200[i], label='JRA-55', color='purple', linewidth=2)
    ax1.plot(np.arange(1979, 2015), amip_100minus200[i].mean(axis=0), label='AMIP', color='dodgerblue', linewidth=2)
    ax1.fill_between(np.arange(1979, 2015), amip_100minus200[i].mean(axis=0) + amip_100minus200[i].std(axis=0),
                        amip_100minus200[i].mean(axis=0) - amip_100minus200[i].std(axis=0),
                        color='dodgerblue', alpha=0.15)
    if i == 0:
        ax1.legend(fontsize=8,markerscale=0.4,framealpha=0.5, loc= 'upper left',frameon=True)
    ax1.set_xticks(np.arange(1979, 2100, 10))
    ax1.margins(x=0)
    ax1.set_ylim((-2,2))
    ax1.set_yticks(np.arange(-2, 2.1, .5))
    ax1.set_xlabel('Year', fontsize=14)
    ax1.set_ylabel('u$_{100}$-u$_{200}$ (m s$^{-1}$)', fontsize=14)
    ax1.tick_params(axis='both', direction='out',which='major', labelsize=12)
    ax1.set_title(title[i], loc='right', fontsize=12)
    ax1.grid(True)
    ax1.text(-0.1, 1.17, sequence[i*2], transform=ax1.transAxes, fontsize=22, fontweight='bold', va='top', ha='left')

    for obs in range(3):
        if obs_p_values[i][obs] < 0.05:
            ax2.scatter(1, obs_trends[i][obs], color=obs_color[obs], label=obs_label[obs], alpha=0.7)
        else:
            ax2.scatter(1, obs_trends[i][obs], color='none', edgecolors=obs_color[obs], label=obs_label[obs], alpha=0.7)
    for model in range(28):
        if p_values_ssp585[i][model] < 0.05:
            ax2.scatter(0, trends_ssp585[i][model]*10, color='r', alpha=0.7)
        else:
            ax2.scatter(0, trends_ssp585[i][model] * 10, color='none', edgecolors='r', alpha=0.7)
    for model in range(9):
        if p_values_amip[i][model] < 0.05:
            ax2.scatter(2, trends_amip[i][model]*10, color='dodgerblue', alpha=0.7)
        else:
            ax2.scatter(2, trends_amip[i][model] * 10, color='none', edgecolors='dodgerblue', alpha=0.7)
    #axes[1].scatter(0, trends_ssp585[model], color='r',label='members', alpha=0.7)
    # Plotting ensemble mean trends as larger, distinct points
    ax2.scatter(0, mean_trend_ssp585[i], color='black', s=80, label='HIST+SSP585', zorder=3)
    ax2.scatter(2, mean_trend_amip[i], marker='^', color='k', s=80, label='AMIP', zorder=3)

    add_bar_labels('r', 0, -.16, mean_trend_ssp585[i])
    add_bar_labels('green', 1, -.19,obs_trends[i][0])
    add_bar_labels( 'blue', 1, -.16,obs_trends[i][1])
    add_bar_labels('purple', 1, -.13,obs_trends[i][2])
    add_bar_labels('dodgerblue', 2, -.16,mean_trend_amip[i])


    # Setting subplot aesthetics
    ax2.set_xlim((-0.4,2.4))
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(scenarios)
    ax2.set_ylim((-0.15,0.55))
    ax2.set_yticks(np.arange(-0.2, 0.6, .1))
    ax2.set_ylabel('upward trend (m s$^{-1}$ decade$^{-1}$)', fontsize=12)
    ax2.tick_params(axis='y', direction='out',which='major', labelsize=12)
    ax2.tick_params(axis='x', direction='out',which='major', labelsize=11)
    ax2.set_title(title[i], loc='right', fontsize=12)
    ax2.text(-0.1, 1.17, sequence[i*2+1], transform=ax2.transAxes, fontsize=22, fontweight='bold', va='top', ha='left')
    if i == 0:
        ax2.legend(fontsize=6,markerscale=0.4,framealpha=0.5, loc='upper right',frameon=True)
################################以上，绘制时间序列和模式间散点##########################################################################################
################################以下，绘制五年平均时间序列和检测归因结果##################################################################################
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import matplotlib.lines as mlines
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator


def plot_time_series(axe, df, signal_names, area, t):
    # 绘制观测信号
    ax = axe
    ax.plot(df['Year'], df['OBS'], color='k', label='OBS', linewidth=2,alpha=1)

    # 绘制强迫信号
    for signal in signal_names:
        if signal in df.columns:
            ax.plot(df['Year'], df[signal], color=signal_colors.get(signal, '#000000'),
                    label=signal, linewidth=2)
    ax.set_ylim([-.7, .7])
    ax.tick_params(axis='x',direction='in', length=3, width=0.6,labelsize=12)  # 设置x轴刻度数字大小
    ax.tick_params(axis='y',direction='in', length=3, width=0.6,labelsize=12)  # 设置y轴刻度数字大小
    # 设置标签和标题
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("u$_{100}$-u$_{200}$ (m s$^{-1}$)", fontsize=14)
    #ax.set_title("Observed and Forced Signals", fontsize=11)
    ax.text(0.97, 1.07, area, fontsize=12, transform=ax.transAxes, va='top', ha='right')
    ax.legend(prop={'size': 8}, loc='best',frameon=False)
    ax.text(-0.05, 1.17, 'c', transform=ax.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')

    # 设置刻度和网格
    ax.set_xticks(t)
    #ax.tick_params(axis='both', direction='in', length=3, width=0.6)

def plot_combined_figure(axes, loc_x1,loc_x2, betaout1, betaout2, signal_names1,signal_names2, area):
    ax1 = axes
    num_signals1 = len(signal_names1)
    #offsets = np.linspace(-1, 1, num_signals1)  # 调整偏移

    # 绘制信号的缩放系数及置信区间
    for i, signal in enumerate(signal_names1):
        beta_low = f'beta1_low'
        beta_hat = f'beta1_hat'
        beta_up = f'beta1_up'
        color = signal_colors[signal]
        offsets = np.arange(-1, 1.1, 1)

        '''ax1.errorbar(
            loc_x1 + offsets[i], betaout1[i][beta_hat],
            yerr=[betaout1[i][beta_hat] - betaout1[i][beta_low], betaout1[i][beta_up] - betaout1[i][beta_hat]],
            fmt='o', color=color, label=signal, capsize=3, linewidth=1.2, markersize=3
        )'''
        # Draw central hollow line and error bars
        ax1.errorbar(
            loc_x1 + offsets[i], betaout1[i][beta_hat],
            yerr=[betaout1[i][beta_hat] - betaout1[i][beta_low], betaout1[i][beta_up] - betaout1[i][beta_hat]],
            fmt='o', color=color, label=signal, capsize=0, elinewidth=8, markersize=0,alpha=.8
        )

        # Create hollow horizontal line at the central value
        ax1.add_line(mlines.Line2D(
            [loc_x1 + offsets[i] - 0.2, loc_x1 + offsets[i] + 0.2],  # Horizontal line length
            [betaout1[i][beta_hat], betaout1[i][beta_hat]], color='white', linewidth=.8, alpha=1
        ))

    num_signals2 = len(signal_names2)
    #offsets = np.linspace(-0.4, 0.4, num_signals2)  # 调整偏移
    offsets = np.arange(-1, 1.1, 1)
    # 绘制信号的缩放系数及置信区间
    for i, signal in enumerate(signal_names2):
        beta_low = f'beta{i + 1}_low'
        beta_hat = f'beta{i + 1}_hat'
        beta_up = f'beta{i + 1}_up'
        color = signal_colors[signal]

        '''ax1.errorbar(
            loc_x2 + offsets[i], betaout2[beta_hat],
            yerr=[betaout2[beta_hat] - betaout2[beta_low], betaout2[beta_up] - betaout2[beta_hat]],
            fmt='o', color=color, label=signal, capsize=3, linewidth=1.2, markersize=3)'''
        # Draw central hollow line and error bars
        ax1.errorbar(
            loc_x2 + offsets[i], betaout2[beta_hat],
            yerr=[betaout2[beta_hat] - betaout2[beta_low], betaout2[beta_up] - betaout2[beta_hat]],
            fmt='o', color=color, label=signal, capsize=0, elinewidth=8, markersize=0,alpha=.8
        )

        # Create hollow horizontal line at the central value
        ax1.add_line(mlines.Line2D(
            [loc_x2 + offsets[i] - 0.2, loc_x2 + offsets[i] + 0.2],  # Horizontal line length
            [betaout2[beta_hat], betaout2[beta_hat]], color='white', linewidth=.8, alpha=1
        ))

    ax1.set_xlim([1,9])
    ax1.xaxis.set_ticks(np.arange(2,9), ['ALL','GHG','AER','','GHG','AER','NAT'], rotation=45,fontsize=12)
    ax1.set_ylim([-8, 8])

    ax1.axhline(0, color='black', linewidth=0.6,alpha=0.3)
    ax1.axhline(1, linestyle='--', color='gray', linewidth=0.6,alpha=0.3)
    #ax1.axhline(-1, linestyle='--', color='gray', linewidth=0.3,alpha=0.3)
    ax1.axvline(5, color='black', linewidth=1, alpha=0.4)
    ax1.set_ylabel("Scaling factors", fontsize=12)
    # 这句命令同时控制x y 轴刻度数字大小
    ax1.yaxis.set_major_locator(MultipleLocator(2))
    ax1.tick_params(axis='y', which='both', direction='in',length=5, width=0.8, bottom=True, labelbottom=True, labelsize=12)
    ax1.tick_params(axis='x', which='both', direction='in',length=5, width=0.8, bottom=True, labelbottom=True, labelsize=12)
    ax1.text(0.16, .98, '1 Signal', fontsize=9, color='grey',transform=ax1.transAxes, va='top')
    ax1.text(0.63, .98, '3 Signals', fontsize=9, color='grey',transform=ax1.transAxes, va='top')
    ax1.text(0.96, 1.07, area, fontsize=12, transform=ax1.transAxes, va='top', ha='right')
    ax1.text(-0.05, 1.17, 'd', transform=ax1.transAxes, fontsize=22, fontweight='bold', va='top', ha='right')


# 调整配色方案
signal_colors = {
    'ALL': '#E31A1C',  # 活力红
    'GHG': '#4DAF4A',   # 浅绿色
    'AER': '#DA70D6',  # 鲜艳紫
    'NAT': '#FF7F00'  # 亮黄色
}

file_path2 = '/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/1980-2019/obs_GHG_aer_nat_ens3run_20_40_N_1980-2019_5yrmean_anom.dat'
file_path1 = '/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/1980-2019/obssigall_20_40_N_1980-2019_5yrmean_anom.dat'
# 读取数据
with open(file_path1, 'r') as f:
    lines = f.readlines()
# 解析数据
Y = list(map(float, lines[0].strip().split()))
X_data_all = [list(map(float, line.strip().split())) for line in lines[1:]]

with open(file_path2, 'r') as f:
    lines = f.readlines()
# 解析数据
X_data_GHGaernat = [list(map(float, line.strip().split())) for line in lines[1:]]
X_data = X_data_all + X_data_GHGaernat

signal_names = ['ALL', 'GHG', 'AER','NAT']
# 将数据转为 DataFrame
# 生成时间序列
t = np.arange(1982, 2022, 5)
data = {'Year': t, 'OBS': Y}
for i, x in enumerate(X_data):
    signal_name = signal_names[i] if i < len(signal_names) else f'X{i + 1}'
    data[signal_name] = x
df = pd.DataFrame(data)

plot_time_series(axe = fig.add_axes(axes[2]), df=df, signal_names = signal_names,
                     area="20-40°N", t = np.arange(1982, 2022, 5))

base_dir = "/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/1980-2019/"
data_files = ["obs_all_20_40_N_1980-2019_sf_RC",'obs_GHG_ens3run_20_40_N_1980-2019_sf_RC',
              'obs_aer_ens3run_20_40_N_1980-2019_sf_RC',
              "obs_GHG_aer_nat_ens3run_20_40_N_1980-2019_sf_RC"]
EOF_single = [2,4,2]
sf_all = []
for i in range(len(data_files)-1):
    file_path1 = os.path.join(base_dir, f"{data_files[i]}.csv")
    df1 = pd.read_csv(file_path1)
    sf_all.append(df1[df1['#EOF'] ==EOF_single[i]])

file_path2 = os.path.join(base_dir, f"{data_files[-1]}.csv")
df2 = pd.read_csv(file_path2)
sf_GHGaernat = df2[df2['#EOF'] == 4]
plot_combined_figure(axes = fig.add_axes(axes[3]),loc_x1=3, loc_x2=7, betaout1 = sf_all,
                     betaout2 = sf_GHGaernat, signal_names1=['ALL','GHG', 'AER'],
                     signal_names2= ['GHG', 'AER','NAT'], area= '20-40°N')
plt.savefig('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/figures/formal_work/timeseries_and_fingerprint_1979-2022.png',dpi=600)
plt.show()
