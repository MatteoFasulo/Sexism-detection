# Sexism Detection
## Introduction
This repository contains the code for Sexism Detection ([**EXIST 2023**](https://www.damianospina.com/publication/plaza-2023-exist/) **task 1**). According to Oxford English Dictionary, sexism is defined as *"prejudice, stereotyping, or discrimination, typically against women, on the basis of sex"*. The task is to classify whether a given text is sexist or not (**sexism detection**). The dataset is provided by the organizers of the EXIST 2023 shared task and labelled by six experts trained to perform this task.

## Evaluation Measures and Baselines
Evaluation is performed using the **standard evaluation metrics** in the context of binary classification tasks: **Accuracy**, **Precision**, **Recall**, **F1-score**.

According to EXIST 2022 for task 1 (sexism detection) the baselines are:
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
The top team was **avacaondata** (EXIST 2022) using an **ensemble** of different **transformers** models:
* BERTweet-large, RoBERTa and DeBERTa v3 for English
* BETO, BERTIN, MarIA-base and RoBERTuito for Spanish
>**Note**: **avacaondata** team ranked fifth for the spanish-only results and first for the multilingual results. The top team for the spanish-only results was **CIMATCOLMEX** using an ensemble of 10 RoBERTuito and 10 BERT models each of them is trained individually using different seeds achieving a +2% improvement over team **avacaondata** in terms of accuracy.

The final submission scores of team **avacaondata** are listed below:

| Language| Accuracy | Precision |  Recall | F1 |
|---------|----------|-----------|---------|----|
| Multilang (best run)| 0.7996   |  0.7982   | 0.7975  | 0.7978 |
| English      | 0.8422   | 0.8388    | 0.8365 | 0.8376 |
| Spanish      | 0.7575   | 0.7574    | 0.7574  | 0.7574 |

## EXIST 2022 vs EXIST 2023
The main difference between EXIST 2022 and EXIST 2023 is that in the latter the data bias that may be introduced both during the data selection and during the labeling process. During the annotation process there is a label bias introduced by socio-demographic differences of the persons that participate in the annotation process, but also when more than one possible correct label exists or when the decision on the label depends on subjectivity.

## Extensions
The task can be extended in order to deal with disagreement among annotators. The learning with disagreement paradigm (LeWiDi) consists mainly in letting systems learn from datasets where no gold annotations are provided but information about the annotations from all annotators, in an attempt to gather the diversity of views (Uma et al.). Rather than eliminating disagreements by selecting the majority vote (EXIST 2021 and 2022), EXIST 2023 will preserve the multiple labels assigned by an heterogeneous and representative group of annotators, so that disagreement can be used as a source of information for the models.

### Dashboard
The dashboard is built using [**Gradio**](https://www.gradio.app/) and the models are hosted on the [**Hugging Face Hub**](https://huggingface.co/). The dashboard is available at the following link: [**Sexism Detection Dashboard**](https://huggingface.co/spaces/MatteoFasulo/Sexism-Detection-Dashboard)
