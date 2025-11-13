"""
评估神经信号频域和时频域保真度的 Python 实现
(已重构以支持多维/多通道信号)

此代码基于以下核心原则：
1. 2.1. 功率谱密度 (PSD) 比较
   - 使用Welch方法进行稳健的PSD估计
   - 比较跨通道平均的对数-对数PSD图
   - 比较跨通道的相对频带功率 (报告 Mean ± Std)
   - 计算跨通道的PSD分布散度 (报告 Mean ± Std)
2. 2.2. 时频分析 (针对非平稳动力学)
   - 使用STFT（频谱图）比较跨通道平均的时频功率

支持的数据格式: (n_trials, n_channels, n_samples)

依赖库:
- numpy
- matplotlib
- scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, stft
from scipy.stats import entropy, wasserstein_distance
from scipy.integrate import simpson as simps  # 用于计算积分（功率）

# 全局小量，防止 log(0) 或除以 0
epsilon = 1e-12

class FrequencyFidelityAssessor:
    """
    一个用于评估真实信号和生成信号之间
    频域及和时频域保真度的类。
    
    此版本设计用于处理多维数据 (n_trials, n_channels, n_samples)。
    """
    
    def __init__(self, real_data, generated_data, fs,
                 spectrogram_path='spectrogram_comparison_multi_dim.png',
                 psd_path='psd_comparison_multi_dim.png'):
        """
        初始化评估器。

        参数:
        real_data (np.array): 真实信号数据，形状 (n_trials, n_channels, n_samples)
        generated_data (np.array): 生成的信号数据，形状 (n_trials, n_channels, n_samples)
        fs (int): 采样频率 (Hz)
        """
        if real_data.ndim != 3 or generated_data.ndim != 3:
            raise ValueError(f"输入数据必须是 3D，格式为 (n_trials, n_channels, n_samples)。"
                             f" 实际得到: {real_data.shape} 和 {generated_data.shape}")
            
        self.real_data = real_data
        self.generated_data = generated_data
        self.fs = fs
        
        self.n_trials_real, self.n_channels_real, self.n_samples_real = real_data.shape
        self.n_trials_gen, self.n_channels_gen, self.n_samples_gen = generated_data.shape

        if self.n_channels_real != self.n_channels_gen:
            print(f"警告: 真实数据和生成数据的通道数不匹配。"
                  f"({self.n_channels_real} vs {self.n_channels_gen})")
            
        # 我们将使用真实数据的通道数进行迭代
        self.n_channels = self.n_channels_real
        
        # 定义经典的神经频带
        self.bands = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 80) # 示例范围
        }

        self.spectrogram_comparison_multi_dim_png_path = spectrogram_path
        self.psd_comparison_multi_dim_png_path = psd_path
        
        print(f"评估器已初始化：")
        print(f"  采样频率: {fs} Hz")
        print(f"  真实数据: {self.n_trials_real} 个试验, {self.n_channels_real} 个通道, {self.n_samples_real} 个样本")
        print(f"  生成数据: {self.n_trials_gen} 个试验, {self.n_channels_gen} 个通道, {self.n_samples_gen} 个样本")

    
    # --- 2.1. 功率谱密度 (PSD) 比较 ---
    
    def _calculate_psd(self, data, nperseg=None):
        """
        (辅助函数) 使用Welch方法计算平均PSD。
        输入数据形状: (n_trials, n_channels, n_samples)
        输出PSD形状: (n_channels, n_freqs) (跨试验平均)
        """
        if nperseg is None:
            # 默认使用2秒的窗口
            nperseg = min(self.fs * 2, data.shape[-1]) 

        all_psds = []
        for i in range(data.shape[0]): # 遍历试验
            trial_data = data[i, :, :] # (n_channels, n_samples)
            
            # Welch 会在倒数第二维(n_channels)上迭代
            # 并在最后一维(n_samples)上计算
            f, Pxx = welch(trial_data, self.fs, nperseg=nperseg, axis=-1)
            # Pxx 的形状是 (n_channels, n_freqs)
            all_psds.append(Pxx)
        
        if not all_psds:
            return None, None
            
        # 对所有试验的PSD取平均
        mean_psd = np.mean(all_psds, axis=0)
        # mean_psd 形状: (n_channels, n_freqs)
        
        # 频率轴应该是相同的
        f, _ = welch(data[0, 0, :], self.fs, nperseg=nperseg, axis=-1) 
        return f, mean_psd

    def plot_psd_comparison(self, nperseg=None):
        """
        [2.1. 视觉检查] 
        在对数-对数坐标上绘制 *跨通道平均* 的PSD。
        (也绘制标准差阴影)
        """
        f_real, psd_real_ch = self._calculate_psd(self.real_data, nperseg)
        f_gen, psd_gen_ch = self._calculate_psd(self.generated_data, nperseg)

        if f_real is None or f_gen is None:
            print("错误：无法计算PSD。")
            return

        # 计算跨通道的平均值和标准差
        psd_real_mean = np.mean(psd_real_ch, axis=0)
        psd_real_std = np.std(psd_real_ch, axis=0)
        
        psd_gen_mean = np.mean(psd_gen_ch, axis=0)
        psd_gen_std = np.std(psd_gen_ch, axis=0)

        plt.figure(figsize=(10, 6))
        
        # 绘制真实数据
        plt.loglog(f_real, psd_real_mean, label='Real Data (Mean across channels)', color='blue', linewidth=2)
        plt.fill_between(f_real, psd_real_mean - psd_real_std, psd_real_mean + psd_real_std, 
                         color='blue', alpha=0.1, label='Real (Std across channels)')
        
        # 绘制生成数据
        plt.loglog(f_gen, psd_gen_mean, label='Generated Data (Mean across channels)', color='red', linestyle='--', linewidth=2)
        plt.fill_between(f_gen, psd_gen_mean - psd_gen_std, psd_gen_mean + psd_gen_std,
                         color='red', alpha=0.1, label='Generated (Std across channels)')
        
        plt.xlabel('Frequency (Hz) [Log Scale]')
        plt.ylabel('Power Spectral Density (V^2/Hz) [Log Scale]')
        plt.title('PSD Comparison (Mean ± Std across Channels)')
        plt.legend()
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.savefig(self.psd_comparison_multi_dim_png_path)

    def _calculate_relative_band_power(self, f, psd_ch):
        """
        (辅助函数) 计算给定PSD的相对频带功率。
        输入 psd_ch 形状: (n_channels, n_freqs)
        输出: 字典, e.g., {'Alpha': array(n_channels,), ...}
        """
        
        # 使用辛普森法则计算总功率 (沿频率轴 axis=1)
        total_power = simps(psd_ch, f, axis=1) # 形状: (n_channels,)
            
        band_powers = {}
        
        for band, (low, high) in self.bands.items():
            # 找到频带对应的频率索引
            idx_band = np.logical_and(f >= low, f <= high)
            if np.sum(idx_band) == 0:
                band_powers[band] = np.zeros(psd_ch.shape[0])
                continue
                
            # 计算该频带的绝对功率
            # psd_ch[:, idx_band] 形状: (n_channels, n_freqs_in_band)
            band_power = simps(psd_ch[:, idx_band], f[idx_band], axis=1) # 形状: (n_channels,)
            
            # 计算相对功率 (逐元素)
            relative_power = band_power / (total_power + epsilon)
            band_powers[band] = relative_power
            
        return band_powers

    def compare_relative_band_power(self, nperseg=None):
        """
        [2.1. 频带功率]
        计算并打印跨通道的 (平均值 ± 标准差) 相对频带功率。
        """
        f_real, psd_real_ch = self._calculate_psd(self.real_data, nperseg)
        f_gen, psd_gen_ch = self._calculate_psd(self.generated_data, nperseg)

        if f_real is None or f_gen is None:
            return None

        # 得到 {band: array(n_channels,)}
        power_real_ch = self._calculate_relative_band_power(f_real, psd_real_ch)
        power_gen_ch = self._calculate_relative_band_power(f_gen, psd_gen_ch)
        
        print("\n--- 相对频带功率比较 (Mean % ± Std across channels) ---")
        print(f"{'Band':<10} | {'Real (Mean ± Std) %':<25} | {'Generated (Mean ± Std) %':<28}")
        print("-" * 70)
        
        results = {}
        for band in self.bands:
            real_mean = np.mean(power_real_ch[band]) * 100
            real_std = np.std(power_real_ch[band]) * 100
            gen_mean = np.mean(power_gen_ch[band]) * 100
            gen_std = np.std(power_gen_ch[band]) * 100
            
            print(f"{band:<10} | {real_mean:>10.2f} ± {real_std:<10.2f} | {gen_mean:>13.2f} ± {gen_std:<10.2f}")
            results[band] = {'real_mean': real_mean, 'real_std': real_std, 'gen_mean': gen_mean, 'gen_std': gen_std}
        
        return results

    def calculate_psd_divergence(self, nperseg=None):
        """
        [2.1. PSD的散度]
        为 *每个通道* 计算散度，然后报告跨通道的 (平均值 ± 标准差)。
        """
        f_real, psd_real_ch = self._calculate_psd(self.real_data, nperseg)
        f_gen, psd_gen_ch = self._calculate_psd(self.generated_data, nperseg)

        if f_real is None or f_gen is None:
            return None

        kl_divs = []
        wasserstein_dists = []

        for i in range(self.n_channels):
            psd_real = psd_real_ch[i, :]
            psd_gen = psd_gen_ch[i, :]

            # 1. 归一化PSD (使其总和为1)
            psd_real_norm = psd_real / (np.sum(psd_real) + epsilon)
            psd_gen_norm = psd_gen / (np.sum(psd_gen) + epsilon)
            
            # 2. KL 散度 (Kullback-Leibler)
            kl_div = entropy(psd_real_norm + epsilon, psd_gen_norm + epsilon)
            kl_divs.append(kl_div)
            
            # 3. Wasserstein 距离 (Earth-Mover's Distance)
            wasserstein_dist = wasserstein_distance(f_real, f_real, psd_real_norm, psd_gen_norm)
            wasserstein_dists.append(wasserstein_dist)
        
        print("\n--- PSD 散度度量 (Mean ± Std across channels) ---")
        print(f"KL Divergence (Real || Gen): {np.mean(kl_divs):.4f} ± {np.std(kl_divs):.4f}")
        print(f"Wasserstein Distance:       {np.mean(wasserstein_dists):.4f} ± {np.std(wasserstein_dists):.4f}")
        
        return (np.mean(kl_divs), np.std(kl_divs)), (np.mean(wasserstein_dists), np.std(wasserstein_dists))

    # --- 2.2. 时频分析 ---

    def plot_spectrogram_comparison(self, nperseg=None):
        """
        [2.2. 短时傅里叶变换 (STFT)]
        计算并绘制 *跨通道平均* 的频谱图。
        """
        if nperseg is None:
            # 时频分析通常需要更短的窗口以获得更好的时间分辨率
            nperseg = self.fs // 2 
            
        noverlap = nperseg // 2 # 50% 重叠

        # 辅助函数：计算平均频谱图
        def _compute_mean_spectrogram(data):
            # data: (n_trials, n_channels, n_samples)
            all_specs = []
            
            # 计算每个试验的STFT
            for i in range(data.shape[0]):
                trial_data = data[i, :, :] # (n_channels, n_samples)
                f, t, Sxx = stft(trial_data, self.fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
                # Sxx 形状: (n_channels, n_freqs, n_times)
                all_specs.append(np.abs(Sxx)) # 使用幅度
            
            # 对所有试验取平均
            # mean_spec 形状: (n_channels, n_freqs, n_times)
            mean_spec_ch = np.mean(all_specs, axis=0)
            
            # f和t对于所有试验都是相同的
            f, t, _ = stft(data[0, 0, :], self.fs, nperseg=nperseg, noverlap=noverlap, axis=-1)
            
            # 返回跨通道平均后的频谱图
            # 形状: (n_freqs, n_times)
            mean_spec_avg_ch = np.mean(mean_spec_ch, axis=0)
            return f, t, mean_spec_avg_ch

        f_real, t_real, spec_real = _compute_mean_spectrogram(self.real_data)
        f_gen, t_gen, spec_gen = _compute_mean_spectrogram(self.generated_data)
        
        # 绘制
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
        
        # 截断高频（通常EEG在80Hz以上信息较少）
        f_max_display = 80
        idx_max = np.where(f_real > f_max_display)[0]
        if len(idx_max) > 0:
            f_idx = idx_max[0]
            f_real = f_real[:f_idx]
            spec_real = spec_real[:f_idx, :]
            f_gen = f_gen[:f_idx]
            spec_gen = spec_gen[:f_idx, :]

        # 转换为dB (10 * log10) 以获得更好的动态范围
        spec_real_db = 10 * np.log10(spec_real + epsilon)
        spec_gen_db = 10 * np.log10(spec_gen + epsilon)
        
        # 绘制真实数据频谱图
        im1 = ax1.pcolormesh(t_real, f_real, spec_real_db, 
                             shading='gouraud', cmap='viridis')
        ax1.set_ylabel('Frequency (Hz)')
        ax1.set_title('Average Spectrogram (Real Data - Avg. Trials & Channels)')
        fig.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # 绘制生成数据频谱图
        im2 = ax2.pcolormesh(t_gen, f_gen, spec_gen_db, 
                             shading='gouraud', cmap='viridis')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Frequency (Hz)')
        ax2.set_title('Average Spectrogram (Generated Data - Avg. Trials & Channels)')
        fig.colorbar(im2, ax=ax2, label='Power (dB)')
        
        plt.tight_layout()
        plt.savefig(self.spectrogram_comparison_multi_dim_png_path)

    def _advanced_analysis_stubs(self):
        """
        [2.2. 小波分析 & HHT]
        为文本中提到的更高级方法提供占位符说明。
        """
        print("\n--- 高级时频分析 (占位符) ---")
        print("提示: 文本中还提到了以下高级方法：")
        print("1. 小波分析 (Wavelet Analysis):")
        print("   - 优势: 相比STFT提供更好的时频分辨率，适合瞬态事件。")
        print("   - 实现: 可使用 `scipy.signal.cwt` 或 `PyWavelets` 库 (需逐通道应用)。")
        print("2. 希尔伯特-黄变换 (HHT):")
        print("   - 优势: 适用于高度非线性和非平稳信号（如EEG）。")
        print("   - 实现: 涉及经验模态分解(EMD)，可使用 `pyhht` 库 (需逐通道应用)。")

    def run_all_analyses(self):
        """
        运行所有已实现的评估。
        """
        print("=====================================================")
        print("### [2.1] 开始功率谱密度 (PSD) 评估 ###")
        print("=====================================================")
        
        print("\n--- 正在生成PSD视觉比较图 (Log-Log, 跨通道 Mean±Std) ---")
        self.plot_psd_comparison()
        
        self.compare_relative_band_power()
        
        self.calculate_psd_divergence()
        
        print("\n=====================================================")
        print("### [2.2] 开始时频分析 (非平稳) 评估 ###")
        print("=====================================================")
        
        print("\n--- G正在生成平均频谱图 (STFT, 跨通道 Mean) 比较 ---")
        self.plot_spectrogram_comparison()
        
        self._advanced_analysis_stubs()


# --- 示例用法 (if __name__ == "__main__") ---

if __name__ == "__main__":
    
    # 1. --- 生成合成的示例数据 (多通道) ---
    
    fs = 250  # 采样频率
    duration = 10  # 信号时长 (秒)
    n_samples = fs * duration
    n_trials = 20
    n_channels = 5 # ! 新增: 5个通道
    t = np.linspace(0, duration, n_samples, endpoint=False)
    
    def create_signal_trials(n_channels, alpha_amp, beta_amp, noise_amp):
        trials = []
        for _ in range(n_trials):
            
            trial_channels = []
            for _ in range(n_channels):
                # 基础Alpha波 (10Hz) - 每个通道有独立相位
                phase_alpha = np.random.rand() * 2 * np.pi
                signal = alpha_amp * np.sin(2 * np.pi * 10 * t + phase_alpha)
                
                # Beta 爆发 (20Hz)，只在 4s 到 6s 之间出现
                if beta_amp > 0:
                    # 每个通道有独立相位
                    phase_beta = np.random.rand() * 2 * np.pi
                    beta_burst = beta_amp * np.sin(2 * np.pi * 20 * t + phase_beta)
                    
                    # 创建一个Hanning窗口作为mask
                    burst_mask = np.zeros(n_samples)
                    start_sample = int(4 * fs)
                    end_sample = int(6 * fs)
                    burst_mask[start_sample:end_sample] = np.hanning(end_sample - start_sample)
                    
                    signal += beta_burst * burst_mask
                    
                # 添加白噪声 (每个通道独立)
                signal += noise_amp * np.random.randn(n_samples)
                trial_channels.append(signal)
            
            trials.append(np.array(trial_channels)) # (n_channels, n_samples)
            
        return np.array(trials) # (n_trials, n_channels, n_samples)

    # 创建真实数据 (有Alpha和Beta爆发)
    real_data = create_signal_trials(n_channels, alpha_amp=1.5, beta_amp=2.0, noise_amp=0.5)
    
    # 创建生成的"失败"数据 (只有Alpha，没有Beta爆发)
    generated_data = create_signal_trials(n_channels, alpha_amp=1.5, beta_amp=0.0, noise_amp=0.5)

    # 2. --- 运行评估 ---
    assessor = FrequencyFidelityAssessor(real_data, generated_data, fs)
    assessor.run_all_analyses()

    # 3. --- 预期结果分析 ---
    #
    # PSD 图:
    # - 曲线将显示跨5个通道的平均PSD。
    # - 阴影区域 (Std) 会显示通道间的一致性。
    # - 真实数据的曲线在 20Hz (Beta) 处会有一个轻微的凸起。
    #
    # 相对频带功率:
    # - Alpha 功率应该接近 (e.g., 45.0 ± 3.0 % vs 44.5 ± 2.8 %)。
    # - 真实数据的 Beta 功率会明显高于生成数据 (e.g., 10.0 ± 2.0 % vs 1.0 ± 0.5 %)。
    #
    # PSD 散度:
    # - 所有指标现在都会有一个平均值和标准差，反映了跨通道的平均保真度和保真度的稳定性。
    #
    # 频谱图 (Spectrogram):
    # - 图像现在是 *跨试验* 和 *跨通道* 的平均图。
    # - 真实数据图: 在 10Hz 处有持续的亮线，*并且* 在 4s-6s 的 20Hz 处有一个明亮的“斑点”。
    # - 生成数据图: 只有 10Hz 处的亮线。
