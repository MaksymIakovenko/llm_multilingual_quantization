import torch
import torch.nn as nn
from tqdm import tqdm

from transformers import AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM
from awq import AutoAWQForCausalLM


# RTN quantization code taken from the orginal AWQ implementation: https://github.com/mit-han-lab/llm-awq/tree/main
def pseudo_quantize_tensor(
    w, n_bit=8, zero_point=True, q_group_size=-1, inplace=False, get_scale_zp=False
):
    org_w_shape = w.shape
    if q_group_size > 0:
        assert org_w_shape[-1] % q_group_size == 0
        w = w.reshape(-1, q_group_size)
    assert w.dim() == 2
    if zero_point:
        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)
    else:  # we actually never used this
        assert min_val is None
        max_val = w.abs().amax(dim=1, keepdim=True)
        max_val = max_val.clamp(min=1e-5)
        max_int = 2 ** (n_bit - 1) - 1
        min_int = -(2 ** (n_bit - 1))
        scales = max_val / max_int
        zeros = 0

    assert torch.isnan(scales).sum() == 0
    assert torch.isnan(w).sum() == 0

    if inplace:
        (
            (w.div_(scales).round_().add_(zeros)).clamp_(min_int, max_int).sub_(zeros)
        ).mul_(scales)
    else:
        w = (
            torch.clamp(torch.round(w / scales) + zeros, min_int, max_int) - zeros
        ) * scales
    assert torch.isnan(w).sum() == 0

    w = w.reshape(org_w_shape)

    if get_scale_zp:
        return w, scales.view(w.shape[0], -1), zeros.view(w.shape[0], -1)
    else:
        return w


def _get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def _get_blocks(model):
    layers = model.model.layers
    return layers


@torch.no_grad()
def pseudo_quantize_model_weight(model, w_bit, q_config):

    layers = _get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = _get_named_linears(layers[i])
        for n, m in named_linears.items():
            m.weight.data = pseudo_quantize_tensor(
                m.weight.data, n_bit=w_bit, **q_config
            )


@torch.no_grad()
def pseudo_quantize_model_weight_targeted(model, w_bit, q_config, targets):

    layers = _get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = _get_named_linears(layers[i])
        for n, m in named_linears.items():
            if m.in_features == 11008:
                # A dirty trick, only down_proj linears have this many in_features
                originals = m.weight.data.detach().clone()
                m.weight.data = pseudo_quantize_tensor(
                    m.weight.data, n_bit=w_bit, **q_config
                )
                m.weight.data[:, targets[i]] = originals[:, targets[i]]
            else:
                # Otherwise quantize regularly
                m.weight.data = pseudo_quantize_tensor(
                    m.weight.data, n_bit=w_bit, **q_config
                )


@torch.no_grad()
def pseudo_quantize_model_weight_targeted_reversed(model, w_bit, q_config, targets):
    # Acts on weights directly responsible for the output of language neurons, rather than the ones that take them is as inputs
    layers = _get_blocks(model)
    for i in tqdm(range(len(layers)), desc="pseudo weight quantization..."):
        named_linears = _get_named_linears(layers[i])
        for n, m in named_linears.items():
            if m.out_features == 11008:
                originals = m.weight.data.detach().clone()
                m.weight.data = pseudo_quantize_tensor(
                    m.weight.data, n_bit=w_bit, **q_config
                )
                m.weight.data[targets[i], :] = originals[targets[i], :]
            else:
                # Otherwise quantize regularly
                m.weight.data = pseudo_quantize_tensor(
                    m.weight.data, n_bit=w_bit, **q_config
                )


def load_model(quant):
    match quant["type"]:
        case "hf":
            if quant["n_bit"] == 4:
                llama = AutoModelForCausalLM.from_pretrained(
                    quant["path"],
                    device_map="auto",
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            else:
                llama = AutoModelForCausalLM.from_pretrained(
                    quant["path"], device_map=None
                )
                llama.to("cuda", dtype=torch.bfloat16)
        case "rtn":
            llama = AutoModelForCausalLM.from_pretrained(quant["path"])
            # for some reason it defaulted to fp32, which makes no sense
            llama.to("cuda", dtype=torch.bfloat16)
            config = {"q_group_size": 128, "inplace": True}
            pseudo_quantize_model_weight(llama, quant["n_bit"], q_config=config)
        case "awq":
            if quant["n_bit"] == 4:
                llama = AutoAWQForCausalLM.from_quantized(
                    quant["path"], device_map="auto", fuse_layers=False
                )
                llama = llama.model
        case "gptq":
            llama = AutoGPTQForCausalLM.from_quantized(quant["path"], device_map="auto")
            llama = llama.model
        case "mpq":
            llama = AutoModelForCausalLM.from_pretrained(quant["path"])
            llama.to("cuda", dtype=torch.bfloat16)
            config = {"q_group_size": 128, "inplace": True}
            neurons = torch.load(quant["neurons_path"])
            pseudo_quantize_model_weight_targeted(
                llama, quant["n_bit"], q_config=config, targets=neurons
            )
        case "mprq":
            llama = AutoModelForCausalLM.from_pretrained(quant["path"])
            llama.to("cuda", dtype=torch.bfloat16)
            config = {"q_group_size": 128, "inplace": True}
            neurons = torch.load(quant["neurons_path"])
            pseudo_quantize_model_weight_targeted_reversed(
                llama, quant["n_bit"], q_config=config, targets=neurons
            )
        case _:
            llama = AutoModelForCausalLM.from_pretrained(
                quant["path"],
                # device_map="auto",
                device_map=None,
            )
            llama.to("cuda", dtype=torch.bfloat16)
    return llama
