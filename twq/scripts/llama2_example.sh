#!/bin/bash

MODEL=llama-2-7b-chat-hf

# run AWQ search (optional; we provided the pre-computed results)
python -m awq.entry --model_path /dataset/llama2-hf/$MODEL \
    --w_bit 3 --q_group_size 128 \
    --run_awq --dump_awq awq_cache/$MODEL-w3-g128.pt

# evaluate the AWQ quantize model (simulated pseudo quantization)
python -m awq.entry --model_path /dataset/llama2-hf/$MODEL \
    --tasks wikitext \
    --w_bit 3 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w3-g128.pt \
    --q_backend fake

# generate real quantized weights (w3)
python -m awq.entry --model_path /dataset/llama2-hf/$MODEL \
    --w_bit 3 --q_group_size 128 \
    --load_awq awq_cache/$MODEL-w3-g128.pt \
    --q_backend real --dump_quant quant_cache/$MODEL-w3-g128-awq.pt

# load and evaluate the real quantized model (smaller gpu memory usage)
python -m awq.entry --model_path /dataset/llama2-hf/$MODEL \
    --tasks wikitext \
    --w_bit 3 --q_group_size 128 \
    --load_quant quant_cache/$MODEL-w3-g128-awq.pt