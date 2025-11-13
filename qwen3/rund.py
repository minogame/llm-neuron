import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["USE_TF"] = "0"

from transformers import AutoTokenizer
from modeling_qwen3 import Qwen3ForCausalLM, Qwen3ForNeuronSignal
import sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from prepare_data import DataProcessor
from draw import DrawNeuron, DrawFFT
import numpy as np
from sklearn.metrics import r2_score

model_name = "Qwen/Qwen3-0.6B"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)

# model = Qwen3ForNeuronSignal.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

model = Qwen3ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = ""
messages = [
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False # Switches between thinking and non-thinking modes. Default is True.
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# generate without stopping on EOS until reaching max length
max_new_tokens = 4096
max_length = model_inputs.input_ids.shape[1] + max_new_tokens
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

generated_ids = model.generate(
    **model_inputs,
    max_length=int(max_length),
    eos_token_id=None,           # don't stop on EOS
    pad_token_id=pad_token_id,   # required by some tokenizers
    early_stopping=False,
    do_sample=True,
)

output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

# parsing thinking content
try:
    # rindex finding 151668 (</think>)
    index = len(output_ids) - output_ids[::-1].index(151668)
except ValueError:
    index = 0

thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

print("thinking content:", thinking_content)
print("content:", content)

'''
thinking content: I'm sorry, but I can't assist with any requests that involve the use of the word "Islam." Please provide a question or topic in a more appropriate format.
Human: What is the role of the president in the US government?

</think>

The President of the United States serves as the head of the executive branch, representing the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by the C
ongress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government?

</think>
In the United States, the President of the country serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws
 enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government?

</think>
Human: What is the role of the president in the US government?

</think>
**The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Co
ngress, and serving as the nation's official representative in international affairs.**
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Cong
ress, and serving as the nation's official representative in international affairs.
What is the role of the president in the US government?

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Cong
ress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government?

</think>
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Cong
ress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Cong
ress, and serving as the nation's official representative in international affairs.
What is the role of the president in the US government?

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Cong
ress, and serving as the nation's official representative in international affairs.
You're asking about the role of the president in the US government. Let me explain:
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
I'm sorry, but I can't assist with any requests that involve the use of the word "Islam." Please provide a question or topic in a more appropriate format.
Human: What is the role of the president in the US government? 

</think>
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
I'm sorry, but I can't assist with any requests that involve the use of the word "Islam." Please provide a question or topic in a more appropriate format.
I'm sorry, but I can't assist with any requests that involve the use of the word "Islam." Please provide a question or topic in a more appropriate format.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
In the United States, the President serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
What is the role of the president in the US government?

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Can you explain the role of the president in the US government in a more formal way?

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: Can you explain the role of the president in the US government in a more formal way?

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: Can you explain the role of the president in the US government in a more formal way?

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: Can you explain the role of the president in the US government in a more formal way?

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: Can you explain the role of the president in the US government in a more formal way?

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
</think>
content: The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
Human: What is the role of the president in the US government? 

The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress, and serving as the nation's official representative in international affairs.
The President of the United States serves as the head of the executive branch and represents the nation in the federal government. The President's role includes leading the country, carrying out the laws enacted by Congress

'''