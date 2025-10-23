import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import math
import pywt
from scipy.signal import find_peaks # 导入 find_peaks 函数
from sklearn.metrics import r2_score
import os
from scipy import signal
from typing import Dict, Tuple, List, Callable, Any, Union

class DrawSVD:

    @staticmethod
    def draw_svd(F, path='outcomes/svd.png'):
        
        U, S, Vt = np.linalg.svd(F, full_matrices=False)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(U[:, 0])
        plt.title('First left singular vector (U1)')
        plt.xlabel('Cell')
        plt.ylabel('Value')

        plt.subplot(1, 3, 2)
        plt.plot(S)
        plt.title('Singular values')
        plt.xlabel('Component')
        plt.ylabel('Value')

        plt.subplot(1, 3, 3)
        plt.plot(Vt[0, :])
        plt.title('First right singular vector (V1)')
        plt.xlabel('Timepoint')
        plt.ylabel('Value')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

        return

class DrawFFT:

    @staticmethod
    def draw_fft_average(F, path='outcomes/fft_average.png'):
        """
        计算并绘制所有细胞的【平均】FFT频谱图。
        （这是您原始代码补全后的版本，为了区分，重命名为 draw_fft_average）
        """
        n_timepoints = F.shape[1]
        if n_timepoints == 0:
            print("Warning: Cannot perform FFT on data with 0 timepoints.")
            return

        fft_magnitude = np.abs(np.fft.fft(F, axis=1))
        mean_fft_magnitude = np.mean(fft_magnitude, axis=0)
        freqs = np.fft.fftfreq(n_timepoints)
        n_positive_freqs = n_timepoints // 2
        
        plt.figure(figsize=(8, 5))
        plt.plot(freqs[:n_positive_freqs], mean_fft_magnitude[:n_positive_freqs])
        plt.title('Average FFT Spectrum Across All Cells')
        plt.xlabel('Frequency')
        plt.ylabel('Average Magnitude')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return

    @staticmethod
    def draw_fft_individual(F, path='outcomes/fft_individual.png', sampling_rate=None):
        """
        在一个大图的N个子图上，为【每一个】细胞绘制其FFT频谱图。

        参数:
        F (np.ndarray): 输入矩阵，形状为 (n_cells, n_timepoints)。
        path (str): 保存输出图像的文件路径。
        sampling_rate (float, optional): 数据的采样率（单位：Hz）。
                                         如果提供，横轴将以赫兹为单位。
                                         如果不提供，横轴将是0到0.5的归一化频率。
        """
        n_cells, n_timepoints = F.shape

        if n_cells == 0:
            print("Warning: Input matrix has 0 cells. No plot will be generated.")
            return

        # 1. 设置子图网格
        ncols = math.ceil(math.sqrt(n_cells))
        nrows = math.ceil(n_cells / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.5), squeeze=False)
        axes_flat = axes.flatten()

        # 2. 计算频率轴
        n_positive_freqs = n_timepoints // 2
        
        if sampling_rate:
            # 如果提供了采样率，计算真实的频率轴（单位：Hz）
            # np.fft.fftfreq 可以直接通过 d=1/sampling_rate 计算出真实频率
            freqs = np.fft.fftfreq(n_timepoints, d=1/sampling_rate)
            xlabel = 'Frequency (Hz)'
        else:
            # 否则，使用归一化频率
            freqs = np.fft.fftfreq(n_timepoints)
            xlabel = 'Normalized Frequency'
        
        # 我们只关心正频率部分
        positive_freqs = freqs[:n_positive_freqs]

        # 3. 遍历每个细胞并绘图
        for i in range(n_cells):
            ax = axes_flat[i]
            cell_fft_magnitude = np.abs(np.fft.fft(F[i, :]))
            
            ax.plot(positive_freqs, cell_fft_magnitude[:n_positive_freqs])
            ax.set_title(f'Cell {i + 1}')
            ax.grid(True)

        # 4. 清理多余的子图
        for i in range(n_cells, len(axes_flat)):
            axes_flat[i].axis('off')

        # 5. 添加总标题和公共坐标轴标签
        fig.suptitle('FFT Spectrum for Each Cell', fontsize=16)
        fig.supxlabel(xlabel) # 使用我们上面定义的 xlabel
        fig.supylabel('Magnitude')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path)
        plt.close(fig)
        return

    @staticmethod
    def draw_fft_stem(F, path='outcomes/fft_stem.png', sampling_rate=None):
        """
        使用【茎图】为每个细胞绘制FFT频谱图，使视觉效果更简洁。
        """
        n_cells, n_timepoints = F.shape
        if n_cells == 0: return

        # --- 网格和频率轴的计算与之前相同 ---
        ncols = math.ceil(math.sqrt(n_cells))
        nrows = math.ceil(n_cells / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
        axes_flat = axes.flatten()

        n_positive_freqs = n_timepoints // 2
        if sampling_rate:
            freqs = np.fft.fftfreq(n_timepoints, d=1/sampling_rate)
            xlabel = 'Frequency (Hz)'
        else:
            freqs = np.fft.fftfreq(n_timepoints)
            xlabel = 'Normalized Frequency'
        positive_freqs = freqs[:n_positive_freqs]

        # --- 核心改动在这里 ---
        for i in range(n_cells):
            ax = axes_flat[i]
            cell_fft_magnitude = np.abs(np.fft.fft(F[i, :]))
            magnitudes_to_plot = cell_fft_magnitude[:n_positive_freqs]

            # 使用 ax.stem() 代替 ax.plot()
            markerline, stemlines, baseline = ax.stem(
                positive_freqs, 
                magnitudes_to_plot,
                linefmt='grey', # 茎的颜色
                markerfmt='o',  # 顶部标记的样式
                basefmt='r-'    # 基线的样式
            )
            # 让样式更简洁
            plt.setp(stemlines, 'linewidth', 1)
            plt.setp(markerline, 'markersize', 4)

            ax.set_title(f'Cell {i + 1}')
            ax.grid(True, linestyle='--', alpha=0.6)

        for i in range(n_cells, len(axes_flat)):
            axes_flat[i].axis('off')

        fig.suptitle('FFT Stem Plot for Each Cell', fontsize=16)
        fig.supxlabel(xlabel)
        fig.supylabel('Magnitude')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path)
        plt.close(fig)
        return        

    @staticmethod
    def draw_fft_peaks(F, path='outcomes/fft_peaks.png', sampling_rate=None, top_k=5):
        """
        寻找每个细胞FFT频谱的【主要峰值】并进行标注，最大程度突出重点。
        """
        n_cells, n_timepoints = F.shape
        if n_cells == 0: return

        ncols = math.ceil(math.sqrt(n_cells))
        nrows = math.ceil(n_cells / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
        axes_flat = axes.flatten()

        n_positive_freqs = n_timepoints // 2
        if sampling_rate:
            freqs = np.fft.fftfreq(n_timepoints, d=1/sampling_rate)
            xlabel = 'Frequency (Hz)'
        else:
            freqs = np.fft.fftfreq(n_timepoints)
            xlabel = 'Normalized Frequency'
        positive_freqs = freqs[:n_positive_freqs]

        for i in range(n_cells):
            ax = axes_flat[i]
            cell_fft_magnitude = np.abs(np.fft.fft(F[i, :]))
            magnitudes_to_plot = cell_fft_magnitude[:n_positive_freqs]

            # --- 核心改动：寻找峰值 ---
            # height=... 设置一个阈值，过滤掉太小的噪声峰值
            # distance=... 设置峰值之间的最小距离，避免在一个大峰上找到多个小峰
            min_height = np.mean(magnitudes_to_plot) # 以平均值为阈值，可自行调整
            peaks, _ = find_peaks(magnitudes_to_plot, height=min_height, distance=5)

            # 如果没有找到峰值，或者为了简洁，只取前 top_k 个最高的峰
            if len(peaks) > 0:
                # 根据峰值幅度排序，取最高的 top_k 个
                top_peak_indices = sorted(peaks, key=lambda p: magnitudes_to_plot[p], reverse=True)[:top_k]
                
                peak_freqs = positive_freqs[top_peak_indices]
                peak_mags = magnitudes_to_plot[top_peak_indices]
                
                # 使用散点图或茎图绘制峰值
                ax.stem(peak_freqs, peak_mags, linefmt='grey', markerfmt='o', basefmt=" ")
                
                # 在峰值旁边标注频率值
                for freq, mag in zip(peak_freqs, peak_mags):
                    ax.text(freq, mag * 1.05, f'{freq:.2f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_title(f'Cell {i + 1} Peaks')
            ax.set_ylim(bottom=0) # 确保y轴从0开始
            ax.set_xlim(left=0, right=positive_freqs[-1]) # 确保x轴范围正确
            ax.grid(True, linestyle='--', alpha=0.6)

        for i in range(n_cells, len(axes_flat)):
            axes_flat[i].axis('off')

        fig.suptitle('Top FFT Peaks for Each Cell', fontsize=16)
        fig.supxlabel(xlabel)
        fig.supylabel('Magnitude')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path)
        plt.close(fig)
        return

class DrawTSNE:
    
    @staticmethod
    def draw_tsne(F, path='outcomes/tsne.png'):
        from sklearn.manifold import TSNE

        tsne = TSNE(n_components=2, random_state=42)
        F_embedded_T = tsne.fit_transform(F.T)
        F_embedded = tsne.fit_transform(F)

        plt.figure(figsize=(16, 6))

        plt.subplot(1, 2, 1)
        idx = np.arange(F_embedded_T.shape[0])
        plt.scatter(F_embedded_T[:, 0], F_embedded_T[:, 1], c=idx, cmap='viridis', s=5, alpha=0.5)
        plt.title('t-SNE of SVD Components (F.T)')

        plt.subplot(1, 2, 2)
        plt.scatter(F_embedded[:, 0], F_embedded[:, 1], s=5, alpha=0.5)
        plt.title('t-SNE of SVD Components (F)')

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

class DrawFull:

    def __init__(self, F):
        self.F = F

    def draw_one_frame(self, frame):
        return self.F[:, frame]

    def roi_size(self, frame):

        s = self.draw_one_frame(frame)
        c = (s + self.F.min() + 10) * 2
        
        c += (s-self.F.min())/self.F.max() * 30

        return c

    def get_alpha(self, frame):
        s = self.draw_one_frame(frame)
        s = s-s.min()
        s = s / s.max()
        s = s * 0.8 + 0.5
        s = np.clip(s, 0.0, 1.0)
        return s

    def update(self, frame):
        self.sc.set_array(self.draw_one_frame(frame))
        self.sc.set_sizes(self.roi_size(frame))
        self.sc.set_alpha(self.get_alpha(frame))
        # Update frame number text
        self.frame_text.set_text(f'{frame}')
        return self.sc, self.frame_text

    def animate(self, all_med_0, all_med_1, path='outcomes/animation.gif'):
        fig, ax = plt.subplots()
        self.sc = ax.scatter(all_med_0, all_med_1, c=self.draw_one_frame(0), s=self.roi_size(0), 
                                cmap='viridis', vmin=-3, vmax=15, alpha=0.5)
        self.sc.set_edgecolor('none')
        plt.colorbar(self.sc, label='Signal Intensity')

        # Add frame number text in red at bottom left
        self.frame_text = ax.text(
            0.01, 0.01, '0', color='red', fontsize=16, ha='left', va='bottom', transform=ax.transAxes
        )

        ani = FuncAnimation(fig, self.update, frames=50, blit=True, interval=100)
        plt.tight_layout()
        ani.save(path, writer='ffmpeg', dpi=72)

class DrawCompare:

    def __init__(self, F, Fp):
        self.F = F
        self.Fp = Fp

    def draw_one_frame(self, frame):
        return self.F[:, frame], self.Fp[:, frame]

    def roi_size(self, frame):

        s, sp = self.draw_one_frame(frame)
        c = (s + self.F.min() + 10) * 2
        cp = (sp + self.F.min() + 10) * 2
        
        c += (s-self.F.min())/self.F.max() * 30
        cp += (sp-self.F.min())/self.F.max() * 30

        return c, cp

    def get_alpha(self, frame):
        s, sp = self.draw_one_frame(frame)
        s = s-s.min()
        sp = sp-sp.min()
        s = s / s.max()
        sp = sp / sp.max()
        s = s * 0.8 + 0.5
        sp = sp * 0.8 + 0.5
        s = np.clip(s, 0.0, 1.0)
        sp = np.clip(sp, 0.0, 1.0)
        return s, sp

    def update(self, frame):
        self.sc.set_array(self.draw_one_frame(frame)[0])
        self.sc.set_sizes(self.roi_size(frame)[0])
        self.sc.set_alpha(self.get_alpha(frame)[0])
        self.scp.set_array(self.draw_one_frame(frame)[1])
        self.scp.set_sizes(self.roi_size(frame)[1])
        self.scp.set_alpha(self.get_alpha(frame)[1])
        # Update frame number text
        self.frame_text.set_text(f'{frame+1515}')
        self.frame_textp.set_text(f'{frame+1515}')
        return self.sc, self.scp, self.frame_text, self.frame_textp

    def animate(self, all_med_0, all_med_1, path='outcomes/animation.gif'):
        # Create two subplots for original and predicted data
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

        self.sc = ax1.scatter(all_med_0, all_med_1, c=self.draw_one_frame(0)[0], s=self.roi_size(0)[0], 
                                cmap='viridis', vmin=-3, vmax=15, alpha=0.5)
        # print(all_med_0.shape, all_med_1.shape, self.draw_one_frame(0)[1].shape, self.roi_size(0)[1].shape)
        # exit()
        self.scp = ax2.scatter(all_med_0, all_med_1, c=self.draw_one_frame(0)[1], s=self.roi_size(0)[1], 
                                cmap='viridis', vmin=-3, vmax=15, alpha=0.5)                        
        self.sc.set_edgecolor('none')
        self.scp.set_edgecolor('none')
        plt.colorbar(self.sc, label='Signal Intensity')
        plt.colorbar(self.scp, label='Predicted Signal Intensity')

        # Add frame number text in red at bottom left
        self.frame_text = ax1.text(
            0.01, 0.01, '1515', color='red', fontsize=16, ha='left', va='bottom', transform=ax1.transAxes
        )
        self.frame_textp = ax2.text(
            0.01, 0.01, '1515', color='red', fontsize=16, ha='left', va='bottom', transform=ax2.transAxes
        )

        ani = FuncAnimation(fig, self.update, frames=500, blit=True, interval=100)
        plt.tight_layout()
        ani.save(path, writer='ffmpeg', dpi=72)

class DrawDiff:
    pass

class DrawNeuron:

    def __init__(self, n_neuron):
        self.n_neuron = n_neuron

    @staticmethod
    def decompose_by_filter(
        data: Union[np.ndarray, List[float]],
        fs: float = 1.0,
        period_cutoffs: Dict[str, float] = None,
        order: int = 5
    ) -> Dict[str, np.ndarray]:
        """Decompose a sequence into long / band / short components (zero-phase filtering)."""
        if period_cutoffs is None:
            period_cutoffs = {'long': 30, 'short': 10}

        arr = np.asarray(data, dtype=float)
        if arr.size < order * 3 + 1:
            raise ValueError(f"Data too short (len={arr.size}) to apply a filter of order={order}.")

        nyq = 0.5 * fs
        wn_long = np.clip((1.0 / period_cutoffs['long']) / nyq, 1e-9, 1 - 1e-9)
        wn_short = np.clip((1.0 / period_cutoffs['short']) / nyq, 1e-9, 1 - 1e-9)
        if wn_long >= wn_short:
            raise ValueError("period_cutoffs['long'] must be greater than period_cutoffs['short'].")

        b_l, a_l = signal.butter(order, wn_long, btype='lowpass')
        b_b, a_b = signal.butter(order, [wn_long, wn_short], btype='bandpass')
        b_s, a_s = signal.butter(order, wn_short, btype='highpass')

        padlen = min(3 * max(len(b_l), len(a_l)), arr.size - 1)
        padlen_arg = padlen if padlen > 0 else 0
        padtype = 'constant' if padlen > 0 else 'odd'

        return {
            'long': signal.filtfilt(b_l, a_l, arr, padlen=padlen_arg, padtype=padtype),
            'band': signal.filtfilt(b_b, a_b, arr, padlen=padlen_arg, padtype=padtype),
            'short': signal.filtfilt(b_s, a_s, arr, padlen=padlen_arg, padtype=padtype),
        }


    def draw_curve(self, x_id, y_id, y_pred, path='neuron_comparison.png'):

        num_channels = self.n_neuron
        x_id = x_id.reshape(self.n_neuron, -1)
        y_id = y_id.reshape(self.n_neuron, -1)
        y_pred = y_pred.reshape(self.n_neuron, -1)
        n_train = x_id.shape[1]

        fig, axes = plt.subplots(num_channels, 1, figsize=(8, 3 * num_channels), sharex=True)

        if num_channels == 1:
            axes = [axes]

        for i in range(num_channels):
            axes[i].plot(x_id[i], label='train', color='blue')
            axes[i].plot(np.arange(n_train, n_train + y_id.shape[1]), y_id[i], label='ground truth', color='orange')
            axes[i].plot(np.arange(n_train, n_train + y_pred.shape[1]), y_pred[i], label='forecast', color='green')
            axes[i].set_title(f'Channel {i}')
            axes[i].legend(loc='lower left')
            # 计算相关系数
            corr = np.corrcoef(y_id[:, i], y_pred[:, i])[0, 1]
            # 计算R^2
            r2 = r2_score(y_id[:, i], y_pred[:, i])
            # 在子图上标注
            axes[i].text(0.02, 0.95, f'Corr={corr:.3f}\nR²={r2:.3f}', transform=axes[i].transAxes,
                            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def draw_spectrum_curve(self, x_id, y_id, y_pred, path='neuron_spectrum_comparison.png'):
        num_channels = self.n_neuron
        x_id = x_id.reshape(self.n_neuron, -1)
        y_id = y_id.reshape(self.n_neuron, -1)
        y_pred = y_pred.reshape(self.n_neuron, -1)
        n_train = x_id.shape[1]

        fig, axes = plt.subplots(num_channels * 4, 1, figsize=(8, 3 * num_channels * 4), sharex=True)

        for i in range(num_channels):
            x_id_y_id = np.concatenate([x_id[i], y_id[i]], axis=0)
            x_id_y_pred = np.concatenate([x_id[i], y_pred[i]], axis=0)

            x_id_y_id_filtered = DrawNeuron.decompose_by_filter(x_id_y_id)
            x_id_y_pred_filtered = DrawNeuron.decompose_by_filter(x_id_y_pred)

            for j in ['full', 'long', 'band', 'short']:
                if j == 'full':
                    axes[i*4].plot(x_id[i], label='train', color='blue')
                    axes[i*4].plot(np.arange(n_train, n_train + y_id.shape[1]), y_id[i], label='ground truth', color='orange')
                    axes[i*4].plot(np.arange(n_train, n_train + y_pred.shape[1]), y_pred[i], label='forecast', color='green')
                    axes[i*4].set_title(f'Channel {i}')
                    axes[i*4].legend(loc='lower left')
                    # 计算相关系数
                    corr = np.corrcoef(y_id[:, i], y_pred[:, i])[0, 1]
                    # 计算R^2
                    r2 = r2_score(y_id[:, i], y_pred[:, i])
                    # 在子图上标注
                    axes[i*4].text(0.02, 0.95, f'Corr={corr:.3f}\nR²={r2:.3f}', transform=axes[i*4].transAxes,
                                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

                else:
                    idx = i * 4 + ['long', 'band', 'short'].index(j) + 1
                    x_id_ = x_id_y_id_filtered[j][:x_id.shape[1]]
                    y_id_ = x_id_y_id_filtered[j][x_id.shape[1]:]
                    y_pred_ = x_id_y_pred_filtered[j][x_id.shape[1]:]

                    axes[idx].plot(x_id_y_id_filtered[j][:x_id.shape[1]], label='train', color='blue')
                    axes[idx].plot(np.arange(n_train, n_train + y_id.shape[1]), x_id_y_id_filtered[j][x_id.shape[1]:], label='ground truth', color='orange')
                    axes[idx].plot(np.arange(n_train, n_train + y_pred.shape[1]), x_id_y_pred_filtered[j][x_id.shape[1]:], label='forecast', color='green')
                    axes[idx].set_title(f'Channel {i} - {j} component')
                    axes[idx].legend(loc='lower left')

                    # 计算相关系数
                    corr = np.corrcoef(y_id_, y_pred_)[0, 1]
                    # 计算R^2
                    r2 = r2_score(y_id_, y_pred_)
                    # 在子图上标注
                    axes[idx].text(0.02, 0.95, f'Corr={corr:.3f}\nR²={r2:.3f}', transform=axes[idx].transAxes,
                                    fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

    def draw_neuron(self, x_id, y_id, y_pred, path='neuron_comparison.png', r2=None, corr=None):
        print(x_id.shape)
        print(y_id.shape)
        print(y_pred.shape)
        x_id = x_id.reshape(self.n_neuron, -1)
        y_id = y_id.reshape(self.n_neuron, -1)
        y_pred = y_pred.reshape(self.n_neuron, -1)
        plt.figure(figsize=(12, 4))
        # 计算所有数据的全局最小最大值
        vmin = min(np.min(x_id), np.min(y_id), np.min(y_pred))
        vmax = max(np.max(x_id), np.max(y_id), np.max(y_pred))

        for i, (data, title) in enumerate(zip([x_id, y_id, y_pred], ['x_id', 'y_id', 'y_pred']), 1):
            ax = plt.subplot(4, 1, i)
            im = ax.imshow(data, aspect='equal', interpolation='nearest', origin='upper', vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.yaxis.set_label_position("left")
            ax.yaxis.tick_left()
            ax.xaxis.set_label_position("bottom")
            ax.xaxis.tick_bottom()
            ax.set_xlim(left=0)
            ax.set_ylim(top=0)
        # 可选：添加一个全局 colorbar
        # plt.colorbar(im, ax=[plt.subplot(4, 1, i) for i in range(1, 4)], orientation='vertical')
        # Optionally, add colorbars if needed:
        # plt.colorbar(im, ax=ax)

        ax = plt.subplot(4, 1, 4)
        ax.axis('off')  # Remove axes and frame
        if r2 is not None and corr is not None:
            ax.text(
            0.5, 0.5,
            f'R2: {r2:.3f}\nCorr: {corr:.3f}',
            color='black',
            fontsize=12,
            ha='center',
            va='center',
            transform=ax.transAxes
            )

        plt.tight_layout()
        plt.savefig(path)
        plt.close()

class DrawWavlet:

    @staticmethod
    def wavelet_denoise(F, path='outcomes/wavelet_denoise_example.png', wavelet = 'db4', level = 5):
        n_points = len(F)
        time = np.linspace(0, n_points/3.4, n_points, endpoint=False)
        signal_noisy = F
        # b. 对带噪信号进行多层小波分解 (Multi-level Decomposition)
        # wavedec 返回的是各层小波系数：[cA5, cD5, cD4, cD3, cD2, cD1]
        # cA 是近似系数（低频），cD 是细节系数（高频）
        coeffs = pywt.wavedec(signal_noisy, wavelet, level=level)

        # c. 对细节系数进行阈值处理（去除噪声）
        # 噪声通常表现为幅度较小的高频系数，我们将其置零
        # 计算一个通用阈值
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(n_points))

        # 将阈值应用于所有细节系数层
        coeffs_thresholded = coeffs.copy()
        for i in range(1, len(coeffs)):
            coeffs_thresholded[i] = pywt.threshold(coeffs[i], threshold, mode='soft')

        # d. 从处理后的系数重构信号 (Reconstruction)
        signal_denoised = pywt.waverec(coeffs_thresholded, wavelet)

        # 确保重构后的信号长度与原始信号一致
        signal_denoised = signal_denoised[:n_points]

        # return signal_denoised
        # print("小波去噪完成。")

        # --- 3. 可视化结果 ---
        plt.figure(figsize=(12, 10))

        plt.subplot(2, 1, 1)
        plt.plot(time, F)
        plt.title('original signal', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.subplot(2, 1, 2)
        plt.plot(time, signal_denoised)
        plt.title(f'({wavelet}) denoised signal', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)

        plt.title('Wavelet denoise')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path)
        plt.close()

    @staticmethod
    def draw_wavelet(F, path='outcomes/wavelet_denoise.png', wavelet='morl', scales=None):
        """
        对每个细胞的信号做连续小波变换（CWT），并绘制其时-频谱图。
        参数:
            F: np.ndarray, 形状为 (n_cells, n_timepoints)
            path: 保存图片路径
            wavelet: 小波类型（如 'morl', 'mexh', 'gaus1' 等）
            scales: 小波尺度列表（如 np.arange(1, 64)），默认自动设置
        """

        n_cells, n_timepoints = F.shape
        if scales is None:
            scales = np.arange(1, min(64, n_timepoints // 2))

        ncols = math.ceil(math.sqrt(n_cells))
        nrows = math.ceil(n_cells / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)
        axes_flat = axes.flatten()

        for i in range(n_cells):
            ax = axes_flat[i]
            signal = F[i, :]
            signal = DrawWavlet.wavelet_denoise(signal)

            coef, freqs = pywt.cwt(signal, scales, wavelet)
            im = ax.imshow(np.abs(coef), aspect='auto', cmap='viridis',
                           extent=[0, n_timepoints, scales[-1], scales[0]])
            ax.set_title(f'Cell {i + 1}')
            ax.set_ylabel('Scale')
            ax.set_xlabel('Time')
            ax.grid(False)

        for i in range(n_cells, len(axes_flat)):
            axes_flat[i].axis('off')

        fig.suptitle('Wavelet Time-Frequency Spectrum for Each Cell', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path)
        plt.close(fig)

from statsmodels.tsa.seasonal import STL
class DrawSTL:
    @staticmethod
    def draw_stl(F, path='outcomes/stl.png', period=None):
        """
        对每个细胞的信号做STL分解，并绘制分解结果（趋势、季节性、残差）。
        参数:
            F: np.ndarray, 形状为 (n_cells, n_timepoints)
            path: 保存图片路径
            period: 周期长度（如24、48等），默认自动估算
        """
        n_cells, n_timepoints = F.shape
        ncols = math.ceil(math.sqrt(n_cells))
        nrows = math.ceil(n_cells / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4), squeeze=False)
        axes_flat = axes.flatten()

        for i in range(n_cells):
            print(f'processing cell = {i}')
            ax = axes_flat[i]
            signal = F[i, :]
            # 自动估算周期
            p = period if period is not None else max(2, n_timepoints // 10)
            try:
                stl = STL(signal, period=p, robust=False)
                res = stl.fit()
                ax.plot(signal, label='Original', color='black', linewidth=1)
                ax.plot(res.trend, label='Trend', color='blue', linewidth=1)
                ax.plot(res.seasonal, label='Seasonal', color='green', linewidth=1)
                ax.plot(res.resid, label='Residual', color='red', linewidth=1)
                ax.set_title(f'Cell {i + 1}')
                ax.legend(fontsize=8)
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', fontsize=10)
                ax.set_title(f'Cell {i + 1} (Error)')
            ax.set_xlabel('Time')
            ax.set_ylabel('Value')
            ax.grid(True, linestyle='--', alpha=0.5)

        for i in range(n_cells, len(axes_flat)):
            axes_flat[i].axis('off')

        fig.suptitle('STL Decomposition for Each Cell', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(path)
        plt.close(fig)