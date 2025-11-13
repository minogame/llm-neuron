import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"

from transformers import AutoTokenizer, AutoModelForCausalLM
from modeling_qwen3 import Qwen3ForCausalLM, Qwen3ForNeuronSignal

import sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from prepare_data import DataProcessor
from draw import DrawNeuron
import numpy as np
from sklearn.metrics import r2_score
from psd_comparison import FrequencyFidelityAssessor

model_size = "0.6B"
model_name = f"Qwen/Qwen3-{model_size}"
os.environ["HF_HOME"] = os.getcwd()

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )
# exit()

model = Qwen3ForNeuronSignal.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

def dump_cuda_memory(tag: str, device):
    if not torch.cuda.is_available():
        return
    device = torch.device(device)
    torch.cuda.synchronize(device)
    fname = f"cuda_mem_{tag}.log"
    with open(fname, "w") as f:
        f.write(torch.cuda.memory_summary(device=device, abbreviated=False))
        f.write("\n\n")
        stats = torch.cuda.memory_stats(device)
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")
    print(f"[memory][{tag}] summary saved to {fname}")

def report_cuda_memory(tag: str, device):
    if not torch.cuda.is_available():
        return
    device = torch.device(device)
    torch.cuda.synchronize(device)
    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    print(f"[memory][{tag}] allocated={allocated:.2f}MB "
          f"reserved={reserved:.2f}MB max_allocated={max_allocated:.2f}MB")
    # dump_cuda_memory(tag, device)
    
# seq_len = model_inputs['input_ids'].shape[1]
# model_inputs['attention_mask'] = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=model.device)).unsqueeze(0).unsqueeze(0)

@torch.no_grad()
def forecast(model, model_inputs, max_length):
    model.eval()
    print(f'{model_inputs["inputs_embeds"].shape = }')
    print(f'{max_length - model_inputs["inputs_embeds"].shape[1] = }')
    print(f'{max_length =  }')
    device = model_inputs["inputs_embeds"].device
    report_cuda_memory("forecast_start", device)
    for step in range(max_length - model_inputs["inputs_embeds"].shape[1]):
        outputs = model(**model_inputs)
        next_token_embeds = outputs.hidden_states[:, -1:, :]
        model_inputs["inputs_embeds"] = torch.cat(
            [model_inputs["inputs_embeds"], next_token_embeds], dim=1
        )
        # model_inputs["past_key_values"] = outputs.past_key_values
        # model_inputs["use_cache"] = True
        # print(model_inputs["inputs_embeds"].shape)

        report_cuda_memory(f"forecast_step_{step + 1}", device)
    report_cuda_memory("forecast_end", device)

    return model_inputs

if __name__ == "__main__":

    device = 'cuda:0'
    n_neuron = model.config.hidden_size
    n_predict = 1024
    n_seq = 4096
    n_frames = n_seq - n_predict

    exp_name = f'qwen3_{model_size}_neuron_signal'
    deco = f'neuron{n_neuron}'

    D = DataProcessor(
        original_data_path='/work03/zhunsun/F_F0_denoisedzsc/Fig5/Fig_5_F_F0_denoisedzsc.npy', 
        iscell_path='/work03/zhunsun/F_F0_denoisedzsc/Fig5/Fig_5_iscell.npy',
        stats_path='/work03/zhunsun/F_F0_denoisedzsc/Fig5/standard_frames_scaled_3056_200924_E235_1_00003_00001_20230331-142029.npy',
        k=n_neuron,)
    
    DW = DrawNeuron(n_neuron=32)
    D.get_max_cov_top_k(source='original')
    data_ = D.max_cov_data.T
    # data_ = D.wavlet_denoise(data_).T
    print(data_.shape)
    print(D.idx)
    # exit()

    model_inputs = {
        "inputs_embeds": torch.tensor(data_[-n_seq:-n_predict], device=device, dtype=torch.bfloat16).unsqueeze(0),
    }

    # outputs = forecast(model, model_inputs, max_length=n_seq)
    # pred = outputs["inputs_embeds"].squeeze(0).to(dtype=torch.float32).cpu().numpy()
    # # normalize pred[-n_predict:] channel-wise
    # for i in range(n_neuron):
    #     pred[-n_predict:, i] = (pred[-n_predict:, i] - np.mean(pred[-n_predict:, i])) / np.std(pred[-n_predict:, i])

    # print(pred.shape)
    # print(pred[-n_predict:])

    # np.save(f'outcomes/{exp_name}_max_cov_pred_{deco}_{n_frames}_{n_predict}.npy', pred)
    # exit()
    pred = np.load(f'outcomes/{exp_name}_max_cov_pred_{deco}_{n_frames}_{n_predict}.npy')

    # for i in range(0, n_neuron, 32):
    #     r2 = r2_score(data_[-n_predict:, i:i+32], pred[-n_predict:, i:i+32])
    #     print(f'Neuron {i}: R2 score = {r2:.4f}')
    #     # compare_fig_path = f'outcomes/{exp_name}_random_compare_{deco}_{n_frames}_{n_predict}_{i}.png'
    #     # DW.draw_neuron(data_[-n_seq:-n_predict, i:i+32].T, data_[-n_predict:, i:i+32].T, pred[-n_predict:, i:i+32].T, path=compare_fig_path)
    #     # curve_fig_path = f'outcomes/{exp_name}_random_curve_{deco}_{n_frames}_{n_predict}_{i}.png'
    #     # DW.draw_curve(data_[-n_seq:-n_predict, i:i+32].T, data_[-n_predict:, i:i+32].T, pred[-n_predict:, i:i+32].T, path=curve_fig_path)
    #     spectrum_fig_path = f'outcomes/{exp_name}_max_cov_spectrum_{deco}_{n_frames}_{n_predict}_{i}_pred_5.png'
    #     DW.draw_spectrum_curve(data_[-n_seq:-n_predict, i:i+32].T, data_[-n_predict:, i:i+32].T, pred[-n_predict:, i:i+32].T, path=spectrum_fig_path)
    #     break

    # original_compare_fig_path = f'outcomes/{exp_name}_original_reconstruct_compare_{deco}_{n_frames}_{n_predict}.png'
    # DW.draw_neuron(D.reconstruct_from_pca(data_[-n_seq:-n_predict]).T, 
    #                D.reconstruct_from_pca(data_[-n_predict:]).T, 
    #                D.reconstruct_from_pca(pred[-n_predict:]).T, path=original_compare_fig_path)


    step = 0
    real_data = np.expand_dims(data_[-n_predict:, :32].T, 0)
    generated_data = np.expand_dims(pred[-n_predict:, :32].T, 0)

    print(f'{real_data.shape = }, {generated_data.shape = }')
    print(real_data)
    # FrequencyFidelityAssessor(real_data=real_data, 
    #                         generated_data=generated_data,
    #                         spectrogram_path=f'outcomes/{exp_name}_spectrogram_{deco}_{step}_{n_frames}_{n_predict}_pred_128.png',
    #                         psd_path=f'outcomes/{exp_name}_psd_{deco}_{step}_{n_frames}_{n_predict}_pred_128.png',
    #                         fs=128).run_all_analyses()

# data_processor.idx = array([2851, 3256, 3103, 1256, 4444, 3114, 2219, 1197, 3205, 2900, 1804,
#        1673, 2998, 2462, 3371, 2487,  724, 3864, 1577, 3148, 2569, 2862,
#        2305, 3153, 2197, 3040, 1269, 2321, 2728, 2958, 3052, 3193])

# 1.7B

# --- 相对频带功率比较 (Mean % ± Std across channels) ---
# Band       | Real (Mean ± Std) %       | Generated (Mean ± Std) %
# ----------------------------------------------------------------------
# Delta      |      41.39 ± 6.70       |         31.39 ± 7.13
# Theta      |      15.90 ± 2.63       |         14.02 ± 4.16
# Alpha      |       9.38 ± 1.74       |         15.74 ± 5.88
# Beta       |      22.01 ± 5.85       |         21.00 ± 7.25
# Gamma      |       2.26 ± 0.77       |          2.18 ± 1.30

# --- PSD 散度度量 (Mean ± Std across channels) ---
# KL Divergence (Real || Gen): 0.3296 ± 0.1188
# Wasserstein Distance:       2.4535 ± 1.5069


#4B
# --- 相对频带功率比较 (Mean % ± Std across channels) ---
# Band       | Real (Mean ± Std) %       | Generated (Mean ± Std) %
# ----------------------------------------------------------------------
# Delta      |      41.39 ± 6.70       |         57.18 ± 9.98
# Theta      |      15.90 ± 2.63       |          1.71 ± 1.13
# Alpha      |       9.38 ± 1.74       |          0.66 ± 0.43
# Beta       |      22.01 ± 5.85       |          0.86 ± 0.66
# Gamma      |       2.26 ± 0.77       |          0.07 ± 0.05

# --- PSD 散度度量 (Mean ± Std across channels) ---
# KL Divergence (Real || Gen): 1.5547 ± 0.5160
# Wasserstein Distance:       6.9080 ± 1.4980

#8B
# --- 相对频带功率比较 (Mean % ± Std across channels) ---
# Band       | Real (Mean ± Std) %       | Generated (Mean ± Std) %
# ----------------------------------------------------------------------
# Delta      |      41.39 ± 6.70       |         36.74 ± 6.22
# Theta      |      15.90 ± 2.63       |          4.00 ± 2.03
# Alpha      |       9.38 ± 1.74       |          5.20 ± 4.75
# Beta       |      22.01 ± 5.85       |          4.16 ± 2.22
# Gamma      |       2.26 ± 0.77       |          0.30 ± 0.17

# --- PSD 散度度量 (Mean ± Std across channels) ---
# KL Divergence (Real || Gen): 1.0546 ± 0.4611
# Wasserstein Distance:       5.7386 ± 1.8619