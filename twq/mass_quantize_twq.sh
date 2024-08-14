model='meta-llama/Llama-2-7b-chat-hf'

echo "TWQ quantization using combined language neurons:"
python -m awq.entry --model_path $model --w_bit 4 --q_group_size 128 --run_awq --dump_awq ../quants/TWQ/llama_7b_twq_4bit_128g_combined/activations.pt --target_neurons ../neurons/combined.neuron.pth
python -m awq.entry --model_path $model --w_bit 4 --q_group_size 128 --dump_fake ../quants/TWQ/llama_7b_twq_4bit_128g_combined --load_awq ../quants/TWQ/llama_7b_twq_4bit_128g_combined/activations.pt --q_backend fake

echo "TWQ quantization using random neurons:"
python -m awq.entry --model_path $model --w_bit 4 --q_group_size 128 --run_awq --dump_awq ../quants/TWQ/llama_7b_twq_4bit_128g_random/activations.pt --target_neurons ../neurons/random.neuron.pth
python -m awq.entry --model_path $model --w_bit 4 --q_group_size 128 --dump_fake ../quants/TWQ/llama_7b_twq_4bit_128g_random --load_awq ../quants/TWQ/llama_7b_twq_4bit_128g_random/activations.pt --q_backend fake