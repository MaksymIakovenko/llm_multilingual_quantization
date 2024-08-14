# Analyzing and Addressing the Impact of Multilingual Data Imbalance in Large Language Model Compression

This repository contains the evaluation code and the relevant results from my masters thesis, which aims to analyze the interaction between modern large language model quantization techniques and the issue of imbalance multilingual performance in these models. 

This file will outline the general structure of the project and provide detailed instructions on how to use the code in this project.

### Abstract

The development and study of Large Language Models (LLMs) is prominent axis of research within the field of artificial intelligence. The state-of-the-art capabilities of these systems in natural language understanding and generation tasks make them powerful productivity tools. Consequently, numerous studies aim to make LLMs faster, less costly, and more accessible to a broader audience.

Weight quantization is a popular family of methods addressing these needs. Quantization aims to reduce a model's memory footprint by decreasing the number of bits required to store each model parameter. In recent years, many effective quantization techniques have been developed that exploit particularities of LLM's inner workings to minimize quality losses associated with precision reduction.

The issue of imbalanced distribution of multilingual data in training corpora remains an understudied area in the context of modern autoregressive LLMs, especially regarding its interaction with contemporary LLM quantization methods. However, recent research has demonstrated that some modern LLMs exhibit an English language bias in their intermediate internal representations.

This thesis explores in great depth the literature relevant to the intersection of these three core research topics, evaluate the performance of the prominent modern LLM quantization methods and proposes potential solutions to alleviate the problem of multilingual data imbalance.

The evaluation results of this study show a systematic link between the prominence of a language in the training corpus of an LLM with the resulting performance of the model on a wide range language manipulation task such as translation and summarisation.

Finally, this work outlines and tests preemptive solutions to alleviate the identified performance degradation through the use of insights gained through the study of the surrounding literature.

## Project Structure

This project is organized as follows:

* The `eval_results` folder aggregates all of the relevant evaluation results from this study 
* The `benchmarks` folder contains the code relevant to the benchmarking of the quantization methods employed in this work
* The `visualize` folder contains a series of jupyter notebooks containing visualization code for the benchmarks
* The `quantize` folder contains scripts used to quantize LLaMA 2 7B Chat 
* The `latent_language_eval` folder contains code relevant to the logit lens evaluation of the models
* The `twq` folder contains the modified version of the AWQ implementation used in this study
* The `neurons` folder contains the neuron indices used for this study
* The `quants` folder is meant to contain the quantized models, which are omitted due to their volume

## Installation

All the code in this project was run on version 3.11.8 of Python, the main dependencies for this project are specified in the `requirements.txt` file and can accordingly be installed in a dedicated environment using:

```bash
pip install -r requirements.txt
```

It is advised to install the dependencies for the `twq` in a separate environment using the separate `requirements.txt` file situated in the `twq` folder to avoid conflicts between the dependencies.

## Usage Guide

This section outlines the general usage guide for various parts of this project

### Model Quantization

To quantize LLaMA 2 7B Chat using the same configuration as in the experiments, simply run the `mass_quantize.sh` script from the `quantize` folder to add quantized models into the `quants` folder. If you do not yet have LLaMA 2 7B Chat downloaded from Hugging Face Hub, consider specifying your API key in the `hf_api_key` field in this script, or by using the `huggingface-cli login` command, as this model is gated behind a license you need to accept.

To quantize LLaMA 2 7B using the TWQ format, run the `mass_quantize_twq.sh` script from the `twq` folder. This script first searches for the optimal scaling factors and then pseudo-quantize the model, both for the language-specific neuron and random neuron configurations.

Keep in mind that the full precision model takes apprimately 14GB of disk space, same goes for the two the pseudo-quantized TWQ models, while the AWQ and GPTQ model will take up approximately 4GB of disk space each.

### Logit Lens Evaluation

The evaluation scripts for logit lens evaluation can be found in the `latent_language_eval` folder. First run the `gen_latents.sh` script to generate and store results for individual tasks separately, after which you can run the `gen_plot.sh` script to produce summary plots which will be stored in the `./eval_results/logit_lens/` sub-directory by default.

### Benchmarking

The FLORES+ BLEU evaluation can be performed by running the `generate_flores_translations.py` script from within the `benchmarks` folder as follows:

```bash
python3 generate_flores_translations.py -c CATEGORY -b BATCH_SIZE -n RUN_NAME
```

The category parameter refer to whether the generic quantization methods should be evaluated, in which case "BASE" should be specified, or whether neuron-targetting prototype quantization methods should be evaluated, in which case "LN" should be specified.

Please download the `v2.0-rc.3` version of the benchmark dataset from the [FLORES+ Github Repository](https://github.com/openlanguagedata/flores). The code assumes that the relevant benchmarks are situated in a `../../Eval/` folder on the same level as the this project's root folder.

The XL-Sum ROUGE evaluation can be performed by running the `generate_xlsum_summaries.py` script from within the `benchmarks` folder as follows:

```bash
python3 generate_xlsum_summaries.py -c CATEGORY -b BATCH_SIZE -n RUN_NAME -s SAMPLE_COUNT -l ALT_PROMPT
```

The category parameter works exactly the same way as for FLORES, the SAMPLE_COUNT refers to the number of samples from the dataset to be used in evaluation, `-l` parameter specifies whether the alternative prompting strategy involving a premade model response beginning should be used (True) or not (False).

Similarly, please make sure to download the XL-Sum benchmark dataset from [XL-Sum Github Repository](https://github.com/csebuetnlp/xl-sum) and put the data for each indvidual language `lang_code` into a `../../Eval/xl-sum/data/[lang_code]/` folder. As an example, for English you would put the `english_test.jsonl`, `english_train.jsonl` and `english_val.jsonl` files into the `../../Eval/xl-sum/data/en/` folder.

The perplexity evaluation is done through the `perplexity.py` and `perplexity_normalized.py` scripts from within the `benchmarks` folder, for regular sliding window perplexity and the language-adjusted approach respectively:

```bash
python3 perplexity.py -c CATEGORY -n RUN_NAME
```

```bash
python3 perplexity_normalized.py -c CATEGORY -n RUN_NAME
```

Much like with the BLEU evaluation on FLORES, these perplexity evaluation scripts require you to have the FLORES+ dataset downloaded and placed in the specified directory.

### Other Relevant Files

Other files of note include:

* `./benchmarks/utils.py`: contains pseudo-quantization functions for round-to-nearest and mixed precision approaches
* `./neurons/generate_neurons.ipynb`: briefly goes over the contents of the language neurons and generates a random set of indeces for the control group
*  `./twq/awq/quantize/auto_scale.py`: contains most of the modifications distinguishing TWQ from AWQ

## Credits

Parts of this project are heavily based on evaluation scripts from ["Do Llamas Work in English? On the Latent Language of Multilingual Transformers"](https://github.com/epfl-dlab/llm-latent-language) and original quantization framework developed for ["AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://github.com/mit-han-lab/llm-awq). Additionally, some of the proposed methods were tested using language-specific neuron indices precomputed as part of ["Language-Specific Neurons: The Key to Multilingual Capabilities in Large Language Models"](https://github.com/RUCAIBox/Language-Specific-Neurons).