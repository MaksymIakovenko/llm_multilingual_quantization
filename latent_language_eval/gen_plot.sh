#!/bin/bash

t_configs=(
    "en fr"
    "en ru"
    "en zh"
    "fr ru"
    "ru zh"
    "zh fr"
)

c_configs=("en" "fr" "ru" "zh")

for t in "${t_configs[@]}"
do
    read -r from to <<< "$t"
    echo "Translating from [$from] to [$to]"
    papermill agregate_plots.ipynb ./visuals/executed_notebooks/translate_plots.ipynb -p input_lang $from -p target_lang $to -p target_task "translation" -p
done

for to in "${c_configs[@]}"
do
    echo "Cloze for [$to]"
    papermill agregate_plots.ipynb ./visuals/executed_notebooks/cloze_plots.ipynb -p target_lang $to -p target_task "cloze"
done

for t in "${t_configs[@]}"
do
    read -r from to <<< "$t"
    echo "Translating from [$from] to [$to]"
    papermill agregate_plots_lang_neurons.ipynb ./visuals/executed_notebooks/translate_plots_ln.ipynb -p input_lang $from -p target_lang $to -p target_task "translation"
done

for to in "${c_configs[@]}"
do
    echo "Cloze for [$to]"
    papermill agregate_plots_lang_neruons.ipynb ./visuals/executed_notebooks/cloze_plots_ln.ipynb -p target_lang $to -p target_task "cloze"
done