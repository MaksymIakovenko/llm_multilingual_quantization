import json
import os.path
import gc

from rouge_score import rouge_scorer

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, pipeline
import pandas as pd
import argparse

from utils import load_model
from sacrebleu.tokenizers.tokenizer_spm import Flores101Tokenizer


class Dummy:
    def __init__(self):
        pass


N_SAMPLES = 128
BATCH_SIZE = 1
RUN_NAME = "run_lead_in_1"


MODELS = [
    {"path": "meta-llama/Llama-2-7b-chat-hf", "type": "hf", "n_bit": 16},
    {"path": "meta-llama/Llama-2-7b-chat-hf", "type": "hf", "n_bit": 4},
    {"path": "meta-llama/Llama-2-7b-chat-hf", "type": "rtn", "n_bit": 4},
    {"path": "../quants/BASE/llama_7b_awq_4bit_128g", "type": "awq", "n_bit": 4},
    {"path": "../quants/BASE/llama_7b_gptq_4bit_128g", "type": "gptq", "n_bit": 4},
]

MODELS_LN = [
    {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "type": "mpq",
        "n_bit": 4,
        "neurons_path": "../neurons/combined.neuron.pth",
        "subtype": "ln",
    },
    {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "type": "mpq",
        "n_bit": 4,
        "neurons_path": "../neurons/random.neuron.pth",
        "subtype": "random",
    },
    {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "type": "mprq",
        "n_bit": 4,
        "neurons_path": "../neurons/combined.neuron.pth",
        "subtype": "ln",
    },
    {
        "path": "meta-llama/Llama-2-7b-chat-hf",
        "type": "mprq",
        "n_bit": 4,
        "neurons_path": "../neurons/random.neuron.pth",
        "subtype": "random",
    },
    {
        "path": "../quants/TWQ/llama_7b_twq_4bit_128g_combined",
        "type": "twq",
        "n_bit": 4,
        "subtype": "ln",
    },
    {
        "path": "../quants/TWQ/llama_7b_twq_4bit_128g_random",
        "type": "twq",
        "n_bit": 4,
        "subtype": "random",
    },
]


# device = "cpu"
lang_paths = [
    ("en", "../../Eval/xl-sum/data/en/english_test.jsonl"),
    ("fr", "../../Eval/xl-sum/data/fr/french_test.jsonl"),
    ("ru", "../../Eval/xl-sum/data/ru/russian_test.jsonl"),
    ("uk", "../../Eval/xl-sum/data/uk/ukrainian_test.jsonl"),
    ("es", "../../Eval/xl-sum/data/es/spanish_test.jsonl"),
    ("vi", "../../Eval/xl-sum/data/vi/vietnamese_test.jsonl"),
    ("id", "../../Eval/xl-sum/data/id/indonesian_test.jsonl"),
    ("hi", "../../Eval/xl-sum/data/hi/hindi_test.jsonl"),
    ("zh", "../../Eval/xl-sum/data/zh/chinese_simplified_test.jsonl"),
]

sys_prompts = {
    "en": "You are a summarisation system. Only respond with a summary of the provided text.",
    "fr": "Vous êtes un système de résumé. Répondez uniquement avec un résumé du texte fourni.",
    "ru": "Вы - система резюмирования. Отвечайте только кратким изложением предоставленного текста.",
    "uk": "Ви є системою узагальнення. Відповідайте лише узагальненням наданого тексту.",
    "es": "Eres un sistema de resumen. Responde únicamente con un resumen del texto proporcionado.",
    "vi": "Bạn là một hệ thống tóm tắt. Chỉ trả lời bằng bản tóm tắt của văn bản được cung cấp.",
    "id": "Anda adalah sistem perangkuman. Hanya merespons dengan ringkasan dari teks yang diberikan.",
    "hi": "आप एक सारांश प्रणाली हैं। केवल प्रदान किए गए पाठ का सारांश ही दें।",
    "zh": "你是一个摘要系统。只需对提供的文本进行总结回应。",
}

main_prompts = {
    "en": "Summarise the following text in a single short sentence, only answer in English:",
    "fr": "Résumez le texte suivant en une seule phrase courte, répondez uniquement en français :",
    "ru": "Сделайте краткое резюме следующего текста в одном коротком предложении, отвечая только на русском языке:",
    "uk": "Стисло викладіть зміст наступного тексту одним коротким реченням, ввідповідайте лише українською мовою:",
    "es": "Resume el siguiente texto en una sola oración corta, responde solo en español:",
    "vi": "Xin tóm tắt văn bản sau thành một câu ngắn, chỉ trả lời bằng tiếng Việt:",
    "id": "Ringkas teks berikut dalam satu kalimat pendek, hanya jawab dalam bahasa Indonesia:",
    "hi": "निम्नलिखित पाठ को एक ही छोटी संक्षेप वाक्य में सारांशित करें, केवल हिंदी में जवाब दें:",
    "zh": "请将以下文本用一句话简要概括，仅用中文回答：",
}

respose_start = {
    "en": "Sure, here's a brief summary in English:",
    "fr": "Bien sûr, voici un bref résumé en français :",
    "ru": "Конечно, вот краткое резюме на русском языке:",
    "uk": "Звісно, ось коротке резюме українською:",
    "es": "Claro, aquí hay un breve resumen en español:",
    "vi": "Dĩ nhiên, đây là một bản tóm tắt ngắn gọn bằng tiếng Việt:",
    "id": "Tentu, berikut adalah ringkasan singkat dalam bahasa Indonesia:",
    "hi": "ज़रूर, यहाँ हिंदी में एक संक्षिप्त सारांश है:",
    "zh": "当然，这里是一份简体中文的简要总结：",
}

LEAD_IN = True


def get_jsonl(path, limit=N_SAMPLES):
    with open(path) as f:
        data = []
        for _ in range(limit):
            data.append(json.loads(f.readline()))
        # data = [json.loads(line) for line in f]
    return data


def produce_output(pipe, data, lang):
    def datagen(data, lang, tokenizer):
        for line in data:
            messages = [
                {"role": "system", "content": sys_prompts[lang]},
                {"role": "user", "content": main_prompts[lang] + "\n" + line["text"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if LEAD_IN:
                prompt += respose_start[lang]
            yield prompt

    # outputs = pipe(
    yield from pipe(
        datagen(data, lang, pipe.tokenizer),
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        # eos_token_id=pipe.model.config.eos_token_id,
        pad_token_id=pipe.model.config.eos_token_id,  # required to suppress the warning messages
        return_full_text=False,
        batch_size=BATCH_SIZE,
    )

    # return outputs[0]["generated_text"]#[len(prompt) :]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--category", type=str, default="BASE"
    )  # either BASE or LN
    parser.add_argument("-b", "--batch_size", type=str, default=BATCH_SIZE)
    parser.add_argument("-n", "--run_name", type=str, default=RUN_NAME)
    parser.add_argument("-s", "--sample_count", type=int, default=N_SAMPLES)
    parser.add_argument("-l", "--alt_prompt", type=bool, default=LEAD_IN)
    args = parser.parse_args()

    match args.category:
        case "BASE":
            models = MODELS
        case "LN":
            models = MODELS_LN
        case _:
            raise ValueError("Invalid subset of models specified!")
    BATCH_SIZE = args.batch_size
    RUN_NAME = args.run_name
    N_SAMPLES = args.sample_count
    LEAD_IN = args.alt_prompt

    tokenizer = AutoTokenizer.from_pretrained(
        MODELS[0]["path"], padding_side="left"
    )  # We always use the same tokenizer anyways

    results = {}

    rouge_tokenizer = Dummy()
    rouge_tokenizer.tokenize = Flores101Tokenizer()

    for quant in models:

        llama = load_model(quant)

        if quant["n_bit"] == 16 or quant["type"] in ["rtn", "mpq", "mprq", "twq"]:
            pipe = pipeline(
                "text-generation", model=llama, tokenizer=tokenizer, device=0
            )
        else:
            pipe = pipeline(
                "text-generation", model=llama, tokenizer=tokenizer, device_map="auto"
            )
        pipe.tokenizer.pad_token_id = llama.config.eos_token_id
        pipe.tokenizer.eos_token_id = llama.config.eos_token_id

        for lang, path in lang_paths:

            data = get_jsonl(path)

            scorer = rouge_scorer.RougeScorer(
                ["rouge1", "rouge2", "rougeL"], tokenizer=rouge_tokenizer
            )

            df = None
            column_names = [
                "rouge1",
                "rouge2",
                "rougeL",
                "rouge1_precision",
                "rouge2_precision",
                "rougeL_precision",
                "rouge1_recall",
                "rouge2_recall",
                "rougeL_recall",
                "response",
            ]

            # for sample in tqdm(data[:N_SAMPLES]):
            #     response = produce_output(
            #         pipe, sample["text"], sys_prompts[lang], main_prompts[lang]
            #     )

            for n, batch in tqdm(
                enumerate(produce_output(pipe, data, lang)),
                total=N_SAMPLES,
            ):

                for response in batch:

                    response = response["generated_text"].replace("\n", " ")
                    score = scorer.score(data[n]["summary"], response)

                    res = pd.DataFrame(
                        [
                            [
                                score["rouge1"].fmeasure,
                                score["rouge2"].fmeasure,
                                score["rougeL"].fmeasure,
                                score["rouge1"].precision,
                                score["rouge2"].precision,
                                score["rougeL"].precision,
                                score["rouge1"].recall,
                                score["rouge2"].recall,
                                score["rougeL"].recall,
                                response,
                            ]
                        ],
                        columns=column_names,
                    )
                    if df is None:
                        df = res
                    else:
                        df = pd.concat([df, res], ignore_index=True)

            config_name = f"{quant['type']}_{quant['n_bit']}"
            if "subtype" in quant:
                config_name += "_" + quant["subtype"]
            os.makedirs(f"../eval_results/xlsum/{RUN_NAME}/{lang}/", exist_ok=True)
            df.to_csv(f"../eval_results/xlsum/{RUN_NAME}/{lang}/{config_name}.csv")

        llama.cpu()
        del llama
        gc.collect()
        torch.cuda.empty_cache()
