import numpy as np
import torch
import torch.nn as nn
from llamawrapper import LlamaHelper, GPTQLlamaHelper, AWQLlamaHelper
from tqdm import tqdm
import gc

# fix random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

DEVICE = "cuda:1"


def token_prefixes(token_str: str):
    n = len(token_str)
    tokens = [token_str[:i] for i in range(1, n + 1)]
    return tokens


def add_spaces(tokens):
    return ["▁" + t for t in tokens] + tokens


def capitalizations(tokens):
    return list(set(tokens))


def unicode_prefix_tokid(zh_char="云", tokenizer=None):
    start = zh_char.encode().__str__()[2:-1].split("\\x")[1]
    unicode_format = "<0x%s>"
    start_key = unicode_format % start.upper()
    if start_key in tokenizer.get_vocab():
        return tokenizer.get_vocab()[start_key]
    return None


def process_tokens(token_str: str, tokenizer, lang):
    with_prefixes = token_prefixes(token_str)
    with_spaces = add_spaces(with_prefixes)
    with_capitalizations = capitalizations(with_spaces)
    final_tokens = []
    for tok in with_capitalizations:
        if tok in tokenizer.get_vocab():
            final_tokens.append(tokenizer.get_vocab()[tok])
    if lang in ["zh", "ru"]:
        tokid = unicode_prefix_tokid(token_str, tokenizer)
        if tokid is not None:
            final_tokens.append(tokid)
    return final_tokens


# id2voc = {id: voc for voc, id in tokenizer.get_vocab().items()}


def get_tokens(token_ids, id2voc):
    return [id2voc[tokid] for tokid in token_ids]


def compute_entropy(probas):
    return (-probas * torch.log2(probas)).sum(dim=-1)


lang2name = {
    "fr": "Français",
    "de": "Deutsch",
    "ru": "Русский",
    "en": "English",
    "zh": "中文",
}


def get_latents(dataset_gap, llama):
    unemb = nn.Sequential(llama.model.model.norm, llama.model.lm_head)
    # prepare for energy plots
    U = list(unemb[1].parameters())[0].detach().cpu().float()
    weights = list(unemb[0].parameters())[0].detach().cpu().float()
    U_weighted = U.clone()
    # U_weighted = U_weighted / ((U_weighted**2).mean(dim=1, keepdim=True))**0.5
    U_weighted *= weights.unsqueeze(0)
    U_normalized = U_weighted / ((U_weighted**2).sum(dim=1, keepdim=True)) ** 0.5
    v = U.shape[0]
    TT = U_normalized.T @ U_normalized
    avgUU = (((U_normalized.T @ U_normalized) ** 2).sum() / v**2) ** 0.5

    latent_token_probs = []
    out_token_probs = []
    entropy = []
    energy = []
    latents_all = []

    for idx, d in tqdm(enumerate(dataset_gap)):
        prompt = d["prompt"]
        latents = llama.latents_all_layers(prompt)
        logits = unemb(latents.to(llama.model.device)).detach().cpu()
        last = logits[:, -1, :].float().softmax(dim=-1).detach().cpu()
        latent_token_probs += [last[:, torch.tensor(d["latent_token_id"])].sum(axis=-1)]
        out_token_probs += [last[:, torch.tensor(d["out_token_id"])].sum(axis=-1)]
        entropy += [compute_entropy(last)]
        latents_all += [latents[:, -1, :].float().detach().cpu().clone()]
        latents_normalized = latents[:, -1, :].float()
        latents_normalized = latents_normalized / (
            ((latents_normalized**2).mean(dim=-1, keepdim=True)) ** 0.5
        )
        latents_normalized /= latents_normalized.norm(dim=-1, keepdim=True)
        norm = ((U_normalized @ latents_normalized.T) ** 2).mean(dim=0) ** 0.5
        energy += [norm / avgUU]

    latent_token_probs = torch.stack(latent_token_probs)
    out_token_probs = torch.stack(out_token_probs)
    entropy = torch.stack(entropy)
    energy = torch.stack(energy)
    latents = torch.stack(latents_all)

    return latent_token_probs, out_token_probs, entropy, energy, latents


def get_latents_from_bnb_model(dataset_gap, model_path, precision, targets=None):
    llama = LlamaHelper(dir=model_path, bnb_quant=precision)
    return get_latents(dataset_gap, llama)


def get_latents_from_hf_model(dataset_gap, model_path, precision, targets=None):
    llama = LlamaHelper(dir=model_path, bnb_quant=16)
    llama.model.to(dtype=torch.bfloat16)
    return get_latents(dataset_gap, llama)


def get_latents_from_gptq_model(dataset_gap, model_path, precision=None, targets=None):
    llama = GPTQLlamaHelper(model_path)
    return get_latents(dataset_gap, llama)


def get_latents_from_awq_model(dataset_gap, model_path, precision=None, targets=None):
    llama = AWQLlamaHelper(model_path)
    return get_latents(dataset_gap, llama)


def get_latents_from_rtn_model(dataset_gap, model_path, precision, targets=None):
    llama = LlamaHelper(dir=model_path, bnb_quant=None, device=DEVICE, device_map=None)
    config = {"q_group_size": 128, "inplace": True}

    pseudo_quantize_model_weight(llama.model, precision, config)
    llama.model.to(DEVICE, dtype=torch.bfloat16)

    return get_latents(dataset_gap, llama)


def get_latents_from_mpq_model(dataset_gap, model_path, precision, targets=None):
    llama = LlamaHelper(dir=model_path, bnb_quant=None, device=DEVICE, device_map=None)
    config = {"q_group_size": 128, "inplace": True}

    pseudo_quantize_model_weight_targeted(llama.model, precision, config, targets)
    # llama.model.model.to(DEVICE, dtype=torch.bfloat16)

    return get_latents(dataset_gap, llama)


def get_latents_from_mpqr_model(dataset_gap, model_path, precision, targets=None):
    llama = LlamaHelper(dir=model_path, bnb_quant=None, device=DEVICE, device_map=None)
    config = {"q_group_size": 128, "inplace": True}

    pseudo_quantize_model_weight_targeted_reversed(
        llama.model, precision, config, targets
    )
    llama.model.to(DEVICE, dtype=torch.bfloat16)

    return get_latents(dataset_gap, llama)


# RTN quantization code shamelessly taken from the orginal AWQ implementation: https://github.com/mit-han-lab/llm-awq/tree/main
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
