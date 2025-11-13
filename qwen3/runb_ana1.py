import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# --- 1. 模拟数据 (请用您的真实数据替换这部分) ---
# 假设:
N_TRIALS = 29       # 试验总数
N_NEURONS = 1024    # 您的神经元数量
N_TIMEPOINTS = 4096 # 每个试验的时间点
SAMPLING_RATE = 1 # 假设的采样率 (Hz) - !! 这对PSD计算至关重要

np_load = np.load("outcomes/runb_hidden_states.npy")  # shape: (N_TRIALS, N_TIMEPOINTS, N_NEURONS)
list_of_trials = [np_load[t, 0] for t in range(N_TRIALS)]  # 用真实数据替换

# --- 2. 核心处理：计算演化矩阵 ---

print("开始计算群体平均PSD演化矩阵...")
all_trial_avg_psds = []
frequencies = None # 用于存储频率轴

# Welch's method 参数 (可调)
# nperseg: 窗口大小，决定频率分辨率。
# 窗口越大，频率分辨率越高，但时间分辨率越低（这里是整个trial，所以还好）
NPERSEG = 512 

for trial_idx, trial_data in enumerate(list_of_trials):
    # trial_data 的 shape 是 (N_TIMEPOINTS, N_NEURONS)
    
    neuron_psds_for_this_trial = []
    
    # 遍历这个 trial 中的所有神经元
    for neuron_idx in range(trial_data.shape[1]):
        neuron_signal = trial_data[:, neuron_idx]
        
        # 使用 Welch 方法计算功率谱密度
        # fs = 采样率
        f, Pxx = signal.welch(neuron_signal, fs=SAMPLING_RATE, nperseg=NPERSEG)
        
        neuron_psds_for_this_trial.append(Pxx)
        
        # 仅在第一次循环时存储频率轴
        if frequencies is None:
            frequencies = f
            
    # 计算这个 trial 的“群体平均PSD”
    # np.stack 将 list 变为 (n_neurons, n_frequencies) 的数组
    # np.mean(..., axis=0) 沿着神经元维度平均
    avg_psd_this_trial = np.mean(np.stack(neuron_psds_for_this_trial, axis=0), axis=0)
    
    all_trial_avg_psds.append(avg_psd_this_trial)

# 将所有 trials 的平均PSD堆叠成一个 2D 矩阵
# 最终 shape: (N_TRIALS, M_FREQUENCIES)
evolution_matrix = np.stack(all_trial_avg_psds, axis=0)

print("演化矩阵计算完毕。Shape:", evolution_matrix.shape)


# --- 3. 可视化：绘制热图 ---

# !! 功率谱通常跨越多个数量级 (1/f 噪声)
# !! 将其转换为 dB (10 * log10) 或至少 log 尺度进行可视化，效果会好得多
evolution_matrix_db = 10 * np.log10(evolution_matrix)
# (如果存在0或负值，使用 np.log(evolution_matrix) 之前要处理)


plt.figure(figsize=(12, 7))

# 我们使用 imshow 来绘制热图
# 我们需要转置矩阵 ( .T )，使其 shape 变为 (M_FREQUENCIES, N_TRIALS)
# 这样频率才能在 Y 轴上，trial 在 X 轴上

# extent=[x_min, x_max, y_min, y_max] 用于正确标记坐标轴
# origin='lower' 确保 (0,0) 在左下角
plt.imshow(
    evolution_matrix_db.T, 
    aspect='auto', 
    origin='lower',
    extent=[0, N_TRIALS, frequencies[0], frequencies[-1]]
)

plt.colorbar(label='Mean Power (dB/Hz)')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Trial Number')
plt.title('Population-Average PSD Evolution Over Trials')

# (可选) 限制Y轴的显示范围，以关注感兴趣的频段
# plt.ylim(0, 40) # 例如，只看 0-40 Hz

plt.savefig('outcomes/population_average_psd_evolution.png', dpi=150, bbox_inches='tight')