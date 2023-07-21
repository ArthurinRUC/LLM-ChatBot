import os
import argparse
import gradio as gr
from typing import Dict, Tuple, Union, Optional

from torch.nn import Module
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM

parser = argparse.ArgumentParser(description="llama Model Conversation")
parser.add_argument("--port", type=int)
parser.add_argument("--gpus", type=int, default=1)
parser.add_argument("--model_dir", type=str, default="", help="pretrained model dir")
parser.add_argument("--peft_dir", type=str, default="", help="peft(e.g. LoRA) model dir")
args = parser.parse_args()
port = args.port
num_gpus = args.gpus
model_dir = args.model_dir
peft_dir = args.peft_dir


def auto_configure_device_map(num_gpus: int) -> Dict[str, int]:
    num_trans_layers = 34
    per_gpu_layers = 36 / num_gpus

    device_map = {
        'transformer.wte': 0,
        'transformer.ln_f': 0,
        'lm_head': 0
    }

    used = 2
    gpu_target = 0
    for i in range(num_trans_layers):
        if used >= per_gpu_layers:
            gpu_target += 1
            used = 0
        assert gpu_target < num_gpus
        device_map[f'transformer.h.{i}'] = gpu_target
        used += 1

    return device_map


def load_model_on_gpus(checkpoint_path: Union[str, os.PathLike], num_gpus: int = 2,
                       device_map: Optional[Dict[str, int]] = None, **kwargs) -> Module:
    if num_gpus < 2 and device_map is None:
        model = LlamaForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half().cuda()
    else:
        from accelerate import dispatch_model

        model = LlamaForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=True, **kwargs).half()

        if device_map is None:
            device_map = auto_configure_device_map(num_gpus)

        model = dispatch_model(model, device_map=device_map)

    return model


print("loading model...")

model = load_model_on_gpus(model_dir, num_gpus)
tokenizer = LlamaTokenizer.from_pretrained(model_dir, trust_remote_code=True)
if peft_dir != "":
    print("loading peft model...")
    from peft import PeftModelForCausalLM 
    model = PeftModelForCausalLM.from_pretrained(model, peft_dir)

model = model.eval()

import torch


old_init = transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__
    
def adaptive_ntk_init(self, dim, max_position_embeddings=2048, base=10000, device=None):
    self.dim = dim
    self.base = base
    old_init(self, dim, max_position_embeddings, base, device)

def adaptive_ntk_forward(self, x, seq_len=None):
    if seq_len > self.max_seq_len_cached:
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        inv_freq = self.inv_freq
        dim = self.dim
        alpha = seq_len / 1024 - 1
        base = self.base * alpha ** (dim / (dim-2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(x.device) / dim ))

        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        cos_cached = emb.cos()[None, None, :, :]
        sin_cached = emb.sin()[None, None, :, :]
        return (
            cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
        )
    return (
        self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype)
    )
transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.forward = adaptive_ntk_forward
transformers.models.llama.modeling_llama.LlamaRotaryEmbedding.__init__ = adaptive_ntk_init


def generate_prompt(instruction):
    return f"""
Below is an instruction that describes a task. Write a response that appropriately completes the request.
{instruction}
 """


def predict(input, max_length, top_p, temperature, history):
    inputs = "".join(["### Instruction:\n" +
                    i[0] +
                    "\n\n" +
                    "### Response: " +
                    i[1] +
                    ("\n\n" if i[1] != "" else "") for i in history + [(input, "")]])
        # if len(input) > max_memory:
        #     input = input[-max_memory:]
    prompt = generate_prompt(inputs)

    generate_config = dict(
        max_length=max_length, 
        do_sample=True,
        top_k=40,
        top_p=top_p, 
        temperature=temperature,
        repetition_penalty=1.02,
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs=inputs.input_ids.cuda(), 
            attention_mask=inputs.attention_mask.cuda(),
            **generate_config
        )
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    history.append((input, response.strip()))

    return str(history), history


def reset_state():
    return []

with gr.Blocks() as demo:
    msg = gr.Textbox()
    run = gr.Button("Submit")
    clear = gr.Button("Clear")
    max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
    top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
    temperature = gr.Slider(0, 1, value=0.95, step=0.01, label="Temperature", interactive=True)
    text = gr.Textbox()
    history = gr.State([])

    run.click(predict, [msg, max_length, top_p, temperature, history], [text, history])
    clear.click(reset_state, [], [history])


demo.queue().launch(server_name="0.0.0.0", server_port=port)
