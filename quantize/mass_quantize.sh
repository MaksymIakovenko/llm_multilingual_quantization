#!/bin/bash

model='meta-llama/Llama-2-7b-chat-hf'
hf_api_key='INSERT YOUR HUGGINGFACE API KEY HERE IF NECESSARY'

for prec in '4'
do 
    echo "gptq $prec-bit base"
    python quantize_gptq.py -s $model -d ../quants/BASE/llama_7b_gptq_${prec}bit_128g -g 128 -b $prec -l en -hf $hf_api_key
done 

for prec in '4'
do 
    echo "awq $prec-bit base"
    python quantize_awq.py -s $model -d ../quants/BASE/llama_7b_awq_${prec}bit_128g -g 128 -b $prec -l en -hf $hf_api_key
done 