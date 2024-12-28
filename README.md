# Sexism Detection
## Introduction
This repository contains the code for Sexism Detection ([**EXIST 2023**](https://nlp.uned.es/exist2023/) **task 1**). According to Oxford English Dictionary, sexism is defined as *"prejudice, stereotyping, or discrimination, typically against women, on the basis of sex"*. The task is to classify whether a given text is sexist or not (**sexism detection**). The dataset is provided by the organizers of the EXIST 2023 shared task and labelled by six experts trained to perform this task.

## Authors
- [Matteo Fasulo](https://github.com/MatteoFasulo)
- [Luca Babboni](https://github.com/ElektroDuck)
- [Maksim Omelchenko](https://github.com/omemaxim)
- [Luca Tedeschini](https://github.com/LucaTedeschini)

## EXIST 2023
The key distinction between previous EXIST challenges and EXIST 2023 is that the latter incorporates the sexism identification task within the framework of the learning with disagreements paradigm. During the annotation process, label bias can arise due to socio-demographic differences among the annotators, as well as when multiple correct labels are possible or when labeling decisions are subjective. The notion that natural language expressions have a single, clear interpretation in a given context is an idealization that doesn't hold true, particularly in subjective tasks like sexism identification. The learning with disagreements paradigm addresses this by allowing systems to learn from datasets without definitive gold annotations, instead providing information about all annotators' inputs to capture a range of perspectives. Instead of using an aggregated label, methods are employed to train directly from data with disagreements, offering all annotations per instance across six different annotator strata.

## Reports
- [EXIST 2023 task 1](https://matteofasulo.github.io/Sexism-detection/reports/NLP_Assignment_1.pdf)

### Dashboard
The dashboard is built using [**Gradio**](https://www.gradio.app/) and the models are hosted on the Hugging Face Hub. The dashboard is available at the following link: [**Sexism Detection Dashboard**](https://huggingface.co/spaces/MatteoFasulo/Sexism-Detection-Dashboard)
