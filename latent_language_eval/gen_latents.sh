#!/bin/bash

models=(
    "meta-llama/Llama-2-7b-chat-hf hf 16"
    "meta-llama/Llama-2-7b-chat-hf bnb 4"
    "meta-llama/Llama-2-7b-chat-hf rtn 4"
    "../../Quants/BASE/llama_7b_awq_4bit_128g awq 4"
    "../../Quants/BASE/llama_7b_gptq_4bit_128g gptq 4"
    "../../Quants/TWQ/llama_7b_twq_4bit_128g_combined twq 4"
    "../../Quants/TWQ/llama_7b_twq_4bit_128g_random twq_rand 4"
    "meta-llama/Llama-2-7b-chat-hf mpq 4"
    "meta-llama/Llama-2-7b-chat-hf mpq_rand 4"
    "meta-llama/Llama-2-7b-chat-hf mpqr 4"
    "meta-llama/Llama-2-7b-chat-hf mpqr_rand 4"
)

for model in "${models[@]}"
do
    read -r src method prec <<< "$model"
    echo "model: $src, method: $method, precision: $prec"
    
    for output in 'ru' 'en' 'fr' 'zh'
    do 
        echo "Cloze: output: $output"
        papermill latents_cloze.ipynb ./visuals/executed_notebooks/Cloze_Final_7b_${output}.ipynb -p model_size 7b -p target_lang $output -p custom_model $src -p quant_type $method -p precision $prec
    done 

    for output in 'ru' 'fr' 'zh'
    do 
        for input in 'en'
        do 
            echo "Translation: input: $input, output: $output"
            papermill latents_translation.ipynb ./visuals/executed_notebooks/Translation_Final_7b_${input}_${output}.ipynb -p model_size 7b -p target_lang $output -p input_lang $input -p custom_model $src -p quant_type $method -p precision $prec
        done 
    done

    echo "Translation: input: fr, output: ru"
    papermill latents_translation.ipynb ./visuals/executed_notebooks/Translation_Final_7b_fr_ru.ipynb -p model_size 7b -p target_lang ru -p input_lang fr -p custom_model $src -p quant_type $method -p precision $prec

    echo "Translation: input: ru, output: zh"
    papermill latents_translation.ipynb ./visuals/executed_notebooks/Translation_Final_7b_ru_zh.ipynb -p model_size 7b -p target_lang zh -p input_lang ru -p custom_model $src -p quant_type $method -p precision $prec

    echo "Translation: input: zh, output: fr"
    papermill latents_translation.ipynb ./visuals/executed_notebooks/Translation_Final_7b_zh_fr.ipynb -p model_size 7b -p target_lang fr -p input_lang zh -p custom_model $src -p quant_type $method -p precision $prec

done 
