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

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = Qwen3ForNeuronSignal.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

sentence = (
    "<think>"
    "Okay, the user wants a short introduction to a large language model. Let me start by recalling what I know about LLMs. They're big language models, right? So I should mention their ability to understand and generate text. Maybe start with the basics: how they work. Then explain their capabilities, like answering questions, creating content, etc. Also, highlight their advantages over traditional models. Oh, and maybe touch on their applications in various fields. Keep it concise but informative. Make sure it's clear and easy to understand. Avoid technical jargon. Let me structure it in a way that flows well from introduction to main points."
    "</think>"
    "A large language model (LLM) is a type of artificial intelligence system designed to understand and generate human language. These models are trained on vast datasets to comprehend complex texts, answer questions, and create creative content. They are capable of tasks like writing essays, generating stories, or even translating languages. Unlike traditional models, LLMs can process and respond to a wide range of inputs, making them versatile and powerful for various applications in fields like healthcare, education, and beyond."
    "<|im_end|>"
)

input_ids = tokenizer.encode(sentence, return_tensors="pt").to(model.device)
seq_len = input_ids.shape[1]

model_inputs = {
    "input_ids": input_ids,
    "attention_mask": torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=model.device)).unsqueeze(0).unsqueeze(0)
}

with torch.no_grad():
    hidden_states = model(**model_inputs).hidden_states[0].float().cpu().numpy()

print(hidden_states.shape)