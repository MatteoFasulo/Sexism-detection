# Sexism detection
## Introduction
This repository contains the code for Sexism identification ([**EXIST 2022**](https://www.damianospina.com/publication/rodriguez-2022-exist/) **task 1**). According to Oxford English Dictionary, sexism is defined as *"prejudice, stereotyping, or discrimination, typically against women, on the basis of sex"*. The task is to classify whether a given text is sexist or not (**sexism identification**). The dataset is provided by the organizers of the EXIST 2022 shared task and labelled by six experts trained to perform this task.

## Evaluation Measures and Baselines
Evaluation is performed using the **standard evaluation metrics** in the context of binary classification tasks: **Accuracy**, **Precision**, **Recall**, **F1-score**.

For task 1 (sexism identification), the following baselines are provided:
- **Support Vector Machine (SVM)** with linear kernel trained on TF-IDF features built from text unigrams.
    * Accuracy: 0.6928
    * Precision: 0.6919
    * Recall: 0.685
    * F1-score: 0.6859
- **Majority Class**
    * Accuracy: 0.5444
    * Precision: 0.5444
    * Recall: 0.5
    * F1-score: 0.3525

## Top Team
The top team was **avacaondata** using an **ensemble** of different **transformers** models:
* BERTweet-large, RoBERTa and DeBERTa v3 for English
* BETO, BERTIN, MarIA-base and RoBERTuito for Spanish
>**Note**: **avacaondata** team ranked fifth for the spanish-only results and first for the multilingual results. The top team for the spanish-only results was **CIMATCOLMEX** using an ensemble of 10 RoBERTuito and 10 BERT models each of them is trained individually using different seeds achieving a +2% improvement over team **avacaondata** in terms of accuracy.

The final submission scores of team **avacaondata** are listed below:

| Language| Accuracy | Precision |  Recall | F1 |
|---------|----------|-----------|---------|----|
| Multilang (best run)| 0.7996   |  0.7982   | 0.7975  | 0.7978 |
| English      | 0.8422   | 0.8388    | 0.8365 | 0.8376 |
| Spanish      | 0.7575   | 0.7574    | 0.7574  | 0.7574 |