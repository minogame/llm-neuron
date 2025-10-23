import numpy as np
import random

from draw import *

class DataProcessor:
    def __init__(self, original_data_path='Fig2/F_F0_denoisedzsc.npy', 
        iscell_path='/work03/zhunsun/mesoscope_spontaneous/figshare_25112837/Fig_2_iscell.npy',
        stats_path='/work03/zhunsun/mesoscope_spontaneous/figshare_25112837/Fig_2_stat.npy',
        N_train=1515,
        k = 32,
        num_bins=32,
        bin_edges=None,
        bin_centers=None,
        actual_num_bins=None):

        self.original_data = np.load(original_data_path)

        try:
            stats = np.load(stats_path, allow_pickle=True)
            iscell = np.load(iscell_path)
            self.all_med_0 = [ stats[i]['med'][0] for i in range(len(stats)) if iscell[i, 0] == 1 ]
            self.all_med_1 = [ stats[i]['med'][1] for i in range(len(stats)) if iscell[i, 0] == 1 ]
        except Exception:
            try:
                stats = stats.item()['stat']
                self.all_med_0 = [ stats[i]['med'][0] for i in range(len(stats)) if iscell[i, 0] == 1 ]
                self.all_med_1 = [ stats[i]['med'][1] for i in range(len(stats)) if iscell[i, 0] == 1 ]
            except:
                raise
                self.all_med_0 = list(range(self.original_data.shape[1]))
                self.all_med_1 = list(range(self.original_data.shape[1]))

        self.full_train = self.original_data[:, :N_train]
        self.full_eval = self.original_data[:, N_train:]
        self.idx = 0
        self.k = k
        self.N_train = N_train

        self.draw = False
        self.num_bins = num_bins
        self.bin_edges = None
        self.bin_centers = None
        self.actual_num_bins = None

        self.train = None
        self.evalu = None
        self.bos_list_train = []
        self.bos_list_evalu = None

        self.bin_edges = bin_edges
        self.bin_centers = bin_centers
        self.actual_num_bins = actual_num_bins

        self.cell_grid_ids = None

    def cellize(self,):

        # 计算棋盘格编号
        grid_size = 8
        all_med_0 = np.array(self.all_med_0)
        all_med_1 = np.array(self.all_med_1)

        # 计算x/y的分界点
        x_bins = np.linspace(all_med_0.min(), all_med_0.max(), grid_size + 1)
        y_bins = np.linspace(all_med_1.min(), all_med_1.max(), grid_size + 1)

        # digitize返回的是1~grid_size，减1变成0~(grid_size-1)
        x_idx = np.digitize(all_med_0, x_bins) - 1
        y_idx = np.digitize(all_med_1, y_bins) - 1

        # 保证索引在合法范围
        x_idx = np.clip(x_idx, 0, grid_size - 1)
        y_idx = np.clip(y_idx, 0, grid_size - 1)

        # 棋盘格编号：row-major顺序
        cell_grid_ids = y_idx * grid_size + x_idx
        self.cell_grid_ids = cell_grid_ids.tolist()

        grid_ids = np.array(self.cell_grid_ids)
        unique_ids, counts = np.unique(grid_ids, return_counts=True)
        # 选择cell数最多的top_k个格子
        top_k_grids = unique_ids[np.argsort(counts)[::-1][:self.k]]

        # 绘制棋盘格，只显示top_k_grids的格子为彩色，其余为黑色
        
        # plt.figure(figsize=(6, 6))
        # colors = np.full_like(cell_grid_ids, fill_value=-1, dtype=int)
        # for i, gid in enumerate(cell_grid_ids):
        #     if gid in top_k_grids:
        #         colors[i] = np.where(top_k_grids == gid)[0][0]
        # # 其余为-1，显示为黑色
        # cmap = plt.get_cmap('tab20', len(top_k_grids))
        # scatter = plt.scatter(all_med_0, all_med_1, c=colors, cmap=cmap, s=10, vmin=0, vmax=len(top_k_grids)-1)
        # # 只为top_k_grids加colorbar
        # plt.colorbar(scatter, label='Top-k Grid ID', ticks=range(len(top_k_grids)))
        # plt.xticks(x_bins)
        # plt.yticks(y_bins)
        # plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        # plt.title('Cell positions on 8x8 grid (top-k colored)')
        # plt.xlabel('X')
        # plt.ylabel('Y')
        # plt.savefig('outcomes/cell_grid.png')
        # plt.close()

    def savenpz(self, path):
        np.savez(path, 
            data_train=self.train, 
            data_evalu=self.evalu, 
            idx=self.idx,
            bin_edges = self.bin_edges,
            bin_centers = self.bin_centers,
            actual_num_bins = self.actual_num_bins)

    def loadnpz(self, path):
        data = np.load(path, allow_pickle=True)
        self.train = data['data_train']
        self.evalu = data['data_evalu']
        self.idx = data['idx']
        self.bin_edges = data.get('bin_edges', None)
        self.bin_centers = data.get('bin_centers', None)
        self.actual_num_bins = data.get('actual_num_bins', None)
        # print(self.bin_edges)
        # print(self.bin_centers)
        # print(self.actual_num_bins)
        # exit()

    def quantize(self, data, num_bins=32):

        if self.bin_edges is None:
            quantiles = np.linspace(0, 100, num_bins + 1)
            bin_edges = np.percentile(data, quantiles)

            self.bin_edges = np.unique(bin_edges)
            self.actual_num_bins = len(self.bin_edges) - 1

            if self.actual_num_bins < num_bins:
                print(f"Warning: Reduced number of bins from {num_bins} to {self.actual_num_bins} due to duplicate edges.")
            if self.actual_num_bins < 1:
                raise ValueError("Cannot create any bins with the given data.")

            self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2

        bin_indices = np.digitize(data, self.bin_edges, right=True)
        bin_indices = np.clip(bin_indices - 1, 0, self.actual_num_bins - 1)
        
        return bin_indices, data

    def quantized_values(self, bin_indices):
        quantized_values = self.bin_centers[bin_indices]

        return quantized_values

    def get_original(self):
        if self.draw:
            DrawSVD.draw_svd(self.original_data, path='outcomes/svd_original.png')
            DrawTSNE.draw_tsne(self.original_data, path='outcomes/tsne_original.png')
        return self.original_data
        
    def get_full(self):
        if self.draw:
            DrawSVD.draw_svd(self.full_train, path='outcomes/svd_full_train.png')
            DrawTSNE.draw_tsne(self.full_train, path='outcomes/tsne_full_train.png')
        return self.full_train, self.full_eval

    def get_top_k_grid_average(self, *args, **kwargs):
        if self.cell_grid_ids is None:
            self.cellize()
        k = self.k
        grid_ids = np.array(self.cell_grid_ids)
        unique_ids, counts = np.unique(grid_ids, return_counts=True)
        # 选择cell数最多的top_k个格子
        top_k_grids = unique_ids[np.argsort(counts)[::-1][:k]]

        # 每个格子的cell索引
        grid_cell_indices = {gid: np.where(grid_ids == gid)[0] for gid in top_k_grids}

        # 对每个格子内所有cell行为求平均
        train = []
        evalu = []
        for gid in top_k_grids:
            idxs = grid_cell_indices[gid]
            train.append(self.original_data[idxs, :self.N_train].mean(axis=0))
            evalu.append(self.original_data[idxs, self.N_train:].mean(axis=0))
        train = np.stack(train, axis=0)
        evalu = np.stack(evalu, axis=0)

        self.idx = top_k_grids
        self.train, self.evalu = train, evalu
        return train, evalu


    def get_max_var_top_k(self, *args, **kwargs):
        k = self.k
        F = self.full_train
        F = F - F.mean(axis=1, keepdims=True)
        F = F / np.std(F, axis=1, keepdims=True)

        # 计算每个axis=0（即每个cell/feature）的方差
        var = np.var(F, axis=1)
        idx = np.argsort(var)[::-1][:k]  # 选择方差最大的前k个
        self.idx = idx
        train, evalu = self.original_data[idx, :self.N_train], self.original_data[idx, self.N_train:]

        if self.draw:
            DrawSVD.draw_svd(train, path=f'outcomes/svd_top_k{k}.png')
            DrawTSNE.draw_tsne(train, path=f'outcomes/tsne_top_k{k}.png')

        self.train, self.evalu = train, evalu
        return train, evalu

    def get_max_cov_top_k(self, *args, **kwargs):
        k = self.k
        F = self.full_train
        F = F - F.mean(axis=1, keepdims=True)
        F = F / np.std(F, axis=1, keepdims=True)

        # 计算协方差矩阵
        cov_matrix = np.cov(F)
        # 对角线元素为自身与自身的协方差，忽略
        # 计算每个cell与其他cell的协方差绝对值之和（或均值）
        cov_sum = np.sum(np.abs(cov_matrix - np.diag(np.diag(cov_matrix))), axis=1)
        idx = np.argsort(cov_sum)[::-1][:k]  # 选择协方差最大的前k个
        self.idx = idx
        train, evalu = self.original_data[idx, :self.N_train], self.original_data[idx, self.N_train:]

        if self.draw:
            DrawSVD.draw_svd(train, path=f'outcomes/svd_cov_top_k{k}.png')
            DrawTSNE.draw_tsne(train, path=f'outcomes/tsne_cov_top_k{k}.png')

        self.train, self.evalu = train, evalu
        return train, evalu

    def k_cluster_centers(self, *args, **kwargs):
        k = self.k
        from sklearn.cluster import KMeans
        from scipy.spatial.distance import cdist

        F = self.full_train

        # 聚类，注意KMeans输入为样本在行，特征在列，这里F.T
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(F.T)

        # 计算每个聚类中心与所有cell的距离，找到距离最近的cell作为代表
        centers = kmeans.cluster_centers_  # shape: (k, n_cells)
        # F.T shape: (n_samples, n_cells)
        # 计算每个中心与所有样本的距离
        distances = cdist(centers, F.T)  # shape: (k, n_samples)
        idx = np.argmin(distances, axis=1)  # 每个中心最近的cell索引（在F.T的行，即原F的axis=0）
        self.idx = idx
        train = self.original_data[idx, :self.N_train]
        evalu = self.original_data[idx, self.N_train:]

        if self.draw:
            DrawSVD.draw_svd(train, path=f'outcomes/svd_cluster_centers_k{k}.png')
            DrawTSNE.draw_tsne(train, path=f'outcomes/tsne_cluster_centers_k{k}.png')

        self.train, self.evalu = train, evalu
        return train, evalu

    def random_k(self, *args, **kwargs):
        k = self.k
        np.random.seed(42)
        idx = np.random.choice(self.original_data.shape[0], size=k, replace=False)
        self.idx = idx
        train = self.original_data[idx, :self.N_train]
        evalu = self.original_data[idx, self.N_train:]

        if self.draw:
            DrawSVD.draw_svd(train, path=f'outcomes/svd_random_k{k}.png')
            DrawTSNE.draw_tsne(train, path=f'outcomes/tsne_random_k{k}.png')

        self.train, self.evalu = train, evalu
        return train, evalu

    def wavlet_denoise(self,):

        def F(signal):
            wavelet = 'db4'
            level = 5
            n_points = len(signal)
            time = np.linspace(0, n_points/3.4, n_points, endpoint=False)
            signal_noisy = signal
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

            return signal_denoised

        for idx in range(self.train.shape[0]):
            self.train[idx] = F(self.train[idx])

        for idx in range(self.evalu.shape[0]):
            self.evalu[idx] = F(self.evalu[idx])


    def yield_one_sample_train(self, frames=16, previous=8, quantize=False):
        
        ## X: |<-------- frames ------->|<------- pad ------->|
        ## Y: |<------- pad ------->|<-------- frames ------->|
        ## M: |<-------- 0 -------->|<---------- 1 ---------->|

        ## A: |1111111111111111111111111|000000000000000000000|
        ## A: |000|111111111111111111111111|000000000000000000|


        if self.train is None or self.evalu is None:
            raise ValueError("Data not prepared. Please run one of the data preparation methods first.")

        if len(self.bos_list_train) == 0:
            total_length = self.train.shape[1]
            self.bos_list_train = list(range(0, total_length - frames))
            random.shuffle(self.bos_list_train)

        data_idx = self.bos_list_train.pop()
        train_sample = self.train[:, data_idx:data_idx + frames + 1]

        x = train_sample[:, :frames]
        y = train_sample[:, 1:]

        loss_mask = np.ones_like(y)
        loss_mask[:, :previous] = 0

        x = x.T.flatten()
        y = y.T.flatten()
        loss_mask = loss_mask.T.flatten()
        
        x_id = None
        y_id = None

        if quantize:
            x_id, x = self.quantize(x, num_bins=self.num_bins)
            y_id, y = self.quantize(y, num_bins=self.num_bins)

        return x_id, y_id, x, y, loss_mask

    def yield_one_sample_evalu(self, frames=16, previous=8, quantize=False):
        """
        Yield one sample from the evaluation set, similar to yield_one_sample_train.
        """
        if self.train is None or self.evalu is None:
            raise ValueError("Data not prepared. Please run one of the data preparation methods first.")

        if self.bos_list_evalu is None:
            self.evalu = np.concatenate([self.train[:,-previous:], self.evalu], axis=1)
            self.bos_list_evalu = []

        if len(self.bos_list_evalu) == 0:
            total_length = self.evalu.shape[1]
            self.bos_list_evalu = list(range(0, total_length - frames))

        data_idx = self.bos_list_evalu.pop(0)
        eval_sample = self.evalu[:, data_idx:data_idx + frames + 1]

        x = eval_sample[:, :frames]
        y = eval_sample[:, 1:]

        loss_mask = np.ones_like(y)
        loss_mask[:, :previous] = 0

        x = x.T.flatten()
        y = y.T.flatten()
        loss_mask = loss_mask.T.flatten()

        x_id = None
        y_id = None

        if quantize:
            x_id, x = self.quantize(x, num_bins=self.num_bins)
            y_id, y = self.quantize(y, num_bins=self.num_bins)

        return x_id, y_id, x, y, loss_mask

    @staticmethod
    def separate_evalu_data(x_id, y_id, x, y, previous=8, n_neuron=32):
        
        return x_id[:previous * n_neuron], y_id, x[:previous * n_neuron], y


if __name__ == "__main__":
    # Example usage
    # 5678, 18252
    # d = DataProcessor(
    #     original_data_path='Fig5/Fig_5_F_F0_denoisedzsc.npy', 
    #     iscell_path='Fig5/Fig_5_iscell.npy',
    #     stats_path='Fig5/standard_frames_scaled_3056_200924_E235_1_00003_00001_20230331-142029.npy',
    #     N_train=18000, k=32)
    n_neuron = 32
    quantize_bins = 32
    # data_processor = DataProcessor(k = n_neuron)
    data_processor = DataProcessor(
        original_data_path='Fig5/Fig_5_F_F0_denoisedzsc.npy', 
        iscell_path='Fig5/Fig_5_iscell.npy',
        stats_path='Fig5/standard_frames_scaled_3056_200924_E235_1_00003_00001_20230331-142029.npy',
        N_train=18000, 
        k=n_neuron,
        num_bins=quantize_bins)

    # data_processor.cellize()
    # data_train, data_evalu = data_processor.get_top_k_grid_average()
    # print(f'{data_train.shape = }')
    # print(f'{data_evalu.shape = }')
    # data_train, data_evalu = data_processor.get_max_cov_top_k(n_neuron)
    # # print(f'{data_train.shape = }')
    # # DrawFFT.draw_fft_peaks(data_train, sampling_rate=3.4, top_k=20)

    # DrawWavlet.wavelet_denoise(data_train[0], path='outcomes/wavelet_denoise_example.png')
    # DrawWavlet.draw_wavelet(data_train)

    # DrawSTL.draw_stl(data_train)

    # d.draw = True
    # d.get_original()

    d = data_processor
    T = DrawFull(d.get_original())
    T.animate(d.all_med_0, d.all_med_1, path='outcomes/fig5_original_animation.gif')

    # d.get_full()
    # d.get_max_var_top_k()
    # d.get_max_cov_top_k()
    # d.k_cluster_centers()
    # d.random_k()
    # d.yield_one_sample_train(frames=4, previous=3, quantize=True)