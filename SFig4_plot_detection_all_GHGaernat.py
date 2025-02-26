import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# 全局设置
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.linewidth'] = 0.6  # 调整坐标轴线宽
plt.rcParams['figure.figsize'] = (3, 4)  # 调整图表尺寸


def plot_time_series(axe, df, signal_names, area, t):
    # 绘制观测信号
    ax = axe
    ax.plot(df['Year'], df['obs'], color='k', label='obs', linewidth=1.5)

    # 绘制强迫信号
    for signal in signal_names:
        if signal in df.columns:
            ax.plot(df['Year'], df[signal], color=signal_colors.get(signal, '#000000'),
                    label=signal, linewidth=1.5)
    ax.set_ylim([-.7, .7])
    ax.tick_params(axis='x', labelsize=7)  # 设置x轴刻度数字大小
    ax.tick_params(axis='y', labelsize=10)  # 设置y轴刻度数字大小
    # 设置标签和标题
    ax.set_xlabel("Year", fontsize=11)
    ax.set_ylabel("u$_{100}$-u$_{200}$ anomaly", fontsize=11)
    #ax.set_title("Observed and Forced Signals", fontsize=11)
    ax.text(0.97, 1.07, area, fontsize=9, transform=ax.transAxes, va='top', ha='right')
    ax.legend(prop={'size': 6}, loc='best')
    ax.text(-0.05, 1.15, 'a', transform=ax.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')

    # 设置刻度和网格
    ax.set_xticks(t)
    ax.tick_params(axis='both', direction='in', length=4, width=0.8)

def plot_combined_figure(axes, loc_x1,loc_x2, betaout1, betaout2, signal_names1,signal_names2, area):
    ax1 = axes
    num_signals1 = len(signal_names1)
    offsets = np.linspace(-0.2, 0.2, num_signals1)  # 调整偏移

    # 绘制信号的缩放系数及置信区间
    for i, signal in enumerate(signal_names1):
        beta_low = f'beta{i + 1}_low'
        beta_hat = f'beta{i + 1}_hat'
        beta_up = f'beta{i + 1}_up'
        color = signal_colors[signal]

        ax1.errorbar(
            loc_x1 + offsets[i], betaout1[beta_hat],
            yerr=[betaout1[beta_hat] - betaout1[beta_low], betaout1[beta_up] - betaout1[beta_hat]],
            fmt='o', color=color, label=signal, capsize=3, linewidth=1.2, markersize=3
        )

    num_signals2 = len(signal_names2)
    offsets = np.linspace(-0.2, 0.2, num_signals2)  # 调整偏移

    # 绘制信号的缩放系数及置信区间
    for i, signal in enumerate(signal_names2):
        beta_low = f'beta{i + 1}_low'
        beta_hat = f'beta{i + 1}_hat'
        beta_up = f'beta{i + 1}_up'
        color = signal_colors[signal]

        ax1.errorbar(
            loc_x2 + offsets[i], betaout2[beta_hat],
            yerr=[betaout2[beta_hat] - betaout2[beta_low], betaout2[beta_up] - betaout2[beta_hat]],
            fmt='o', color=color, label=signal, capsize=3, linewidth=1.2, markersize=3)
    #ax1.xaxis.set_ticks(np.arange(4), ['', '1 signal', '3 signals', ''])
    ax1.set_xlim([1,4])
    ax1.xaxis.set_ticks(np.arange(2,4), ['1 signal','3 signals'], rotation=45)
    #ax.xaxis.set_ticks(np.arange(1, len(model_index) * 2 + 2, 2), models, fontsize=6, rotation=90)

    ax1.set_ylim([-2, 8])
    ax1.axhline(0, color='black', linewidth=0.6)
    ax1.axhline(1, linestyle='--', color='gray', linewidth=0.6)
    ax1.axhline(-1, linestyle='--', color='gray', linewidth=0.6)
    ax1.set_ylabel("Scaling factors", fontsize=10)
    # 这句命令同时控制x y 轴刻度数字大小
    ax1.tick_params(axis='both', which='both', direction='in', bottom=True, labelbottom=True, labelsize=10)
    ax1.legend(fontsize=7, loc='best')
    ax1.text(0.96, 1.07, area, fontsize=9, transform=ax1.transAxes, va='top', ha='right')
    ax1.text(-0.05, 1.15, 'b', transform=ax1.transAxes, fontsize=15, fontweight='bold', va='top', ha='right')


# 调整配色方案
signal_colors = {
    'all': '#E31A1C',  # 活力红
    'GHG': '#4DAF4A',   # 浅绿色
    'aer': '#DA70D6',  # 鲜艳紫
    'nat': '#FF7F00'  # 亮黄色
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

signal_names = ['all', 'GHG', 'aer','nat']
# 将数据转为 DataFrame
# 生成时间序列
t = np.arange(1980, 2020, 5)
data = {'Year': t, 'obs': Y}
for i, x in enumerate(X_data):
    signal_name = signal_names[i] if i < len(signal_names) else f'X{i + 1}'
    data[signal_name] = x
df = pd.DataFrame(data)
# 绘制时间序列图
fig = plt.figure(figsize=(6, 3), dpi=500)
sub_add_axes = [[0.1, 0.2, 0.5, 0.6], [0.7, 0.2, 0.25, 0.6]]
plot_time_series(axe = fig.add_axes(sub_add_axes[0]), df=df, signal_names = signal_names,
                     area="20-40°N", t = np.arange(1980, 2020, 5))


base_dir = "/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/fingerprint/1980-2019/"
data_files = ["obs_all_20_40_N_1980-2019_sf_RC", "obs_GHG_aer_nat_ens3run_20_40_N_1980-2019_sf_RC"]
#data_files = ["data_N_all"]
file_path1 = os.path.join(base_dir, f"{data_files[0]}.csv")
df1 = pd.read_csv(file_path1)
sf_all = df1[df1['#EOF'] ==4]
'''plot_combined_figure(axes = fig.add_axes(sub_add_axes[1]),loc_x=1, betaout = sf_all,
                     signal_names= ['all'], area= '20-40°N')'''


file_path2 = os.path.join(base_dir, f"{data_files[1]}.csv")
df2 = pd.read_csv(file_path2)
sf_GHGaernat = df2[df2['#EOF'] == 4]
plot_combined_figure(axes = fig.add_axes(sub_add_axes[1]),loc_x1=2, loc_x2=3, betaout1 = sf_all,
                     betaout2 = sf_GHGaernat, signal_names1=['all'],
                     signal_names2= ['GHG', 'aer','nat'], area= '20-40°N')
plt.savefig('/home/dongyl/UPWARD_SHIFT_OF_JET_STREAM_DATAFILES/figures/formal_work/fingerprint_all_GHGaernat_1958-2017.png',dpi=600)
plt.show()

