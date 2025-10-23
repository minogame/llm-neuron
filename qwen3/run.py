import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from transformers import AutoTokenizer
from modeling_qwen3 import Qwen3ForCausalLM, Qwen3ForNeuronSignal
import sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from prepare_data import DataProcessor
from draw import DrawNeuron
import numpy as np
from sklearn.metrics import r2_score

model_name = "Qwen/Qwen3-0.6B"
os.environ["HF_HOME"] = os.getcwd()

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
    n_neuron = 1024
    n_predict = 256
    n_seq = 4096
    n_frames = n_seq - n_predict

    exp_name = f'qwen3_neuron_signal'
    deco = f'neuron{n_neuron}'

    D = DataProcessor(
        original_data_path='/work03/zhunsun/F_F0_denoisedzsc/Fig5/Fig_5_F_F0_denoisedzsc.npy', 
        iscell_path='/work03/zhunsun/F_F0_denoisedzsc/Fig5/Fig_5_iscell.npy',
        stats_path='/work03/zhunsun/F_F0_denoisedzsc/Fig5/standard_frames_scaled_3056_200924_E235_1_00003_00001_20230331-142029.npy',
        k=n_neuron,)
    
    DW = DrawNeuron(n_neuron=32)
    D.random_k()
    data_ = D.random_data.T
    print(data_.shape)

    model_inputs = {
        "inputs_embeds": torch.tensor(data_[-n_seq:-n_predict], device=device, dtype=torch.bfloat16).unsqueeze(0),
    }

    # outputs = forecast(model, model_inputs, max_length=n_seq)

    # with torch.profiler.profile(
    #     activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    #     profile_memory=True,
    #     record_shapes=True) as prof:
    #     outputs = forecast(model, model_inputs, max_length=n_seq)
    # print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=20))

    # pred = outputs["inputs_embeds"].squeeze(0).to(dtype=torch.float32).cpu().numpy()
    pred = np.load(f'outcomes/{exp_name}_random_pred_{deco}_{n_frames}_{n_predict}.npy')
    # normalize pred[-n_predict:] channel-wise
    for i in range(n_neuron):
         pred[-n_predict:, i] = (pred[-n_predict:, i] - np.mean(pred[-n_predict:, i])) / np.std(pred[-n_predict:, i])

    print(pred.shape)
    print(pred[-n_predict:])
    # np.save(f'outcomes/{exp_name}_random_pred_{deco}_{n_frames}_{n_predict}.npy', pred)

    for i in range(0, n_neuron, 32):
        r2 = r2_score(data_[-n_predict:, i:i+32], pred[-n_predict:, i:i+32])
        print(f'Neuron {i}: R2 score = {r2:.4f}')
        # compare_fig_path = f'outcomes/{exp_name}_random_compare_{deco}_{n_frames}_{n_predict}_{i}.png'
        # DW.draw_neuron(data_[-n_seq:-n_predict, i:i+32].T, data_[-n_predict:, i:i+32].T, pred[-n_predict:, i:i+32].T, path=compare_fig_path)
        # curve_fig_path = f'outcomes/{exp_name}_random_curve_{deco}_{n_frames}_{n_predict}_{i}.png'
        # DW.draw_curve(data_[-n_seq:-n_predict, i:i+32].T, data_[-n_predict:, i:i+32].T, pred[-n_predict:, i:i+32].T, path=curve_fig_path)
        spectrum_fig_path = f'outcomes/{exp_name}_random_spectrum_{deco}_{n_frames}_{n_predict}_{i}.png'
        DW.draw_spectrum_curve(data_[-n_seq:-n_predict, i:i+32].T, data_[-n_predict:, i:i+32].T, pred[-n_predict:, i:i+32].T, path=spectrum_fig_path)

    
    # original_compare_fig_path = f'outcomes/{exp_name}_original_reconstruct_compare_{deco}_{n_frames}_{n_predict}.png'
    # DW.draw_neuron(D.reconstruct_from_pca(data_[-n_seq:-n_predict]).T, 
    #                D.reconstruct_from_pca(data_[-n_predict:]).T, 
    #                D.reconstruct_from_pca(pred[-n_predict:]).T, path=original_compare_fig_path)

