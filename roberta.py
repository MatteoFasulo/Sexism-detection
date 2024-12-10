import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, PrecisionRecallDisplay
import torch
from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

class roBERTa:
    def __init__(self, model_card: str, class_weights: torch.Tensor, seed: int):
        self.model_card = model_card
        self.class_weights = class_weights
        self.seed = seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear the cache and perform garbage collection on initialization
        self.clear_cache()
    
    def clear_cache(self):
        """
        Clears the cache for CUDA and performs garbage collection.

        This method uses PyTorch's `torch.cuda.empty_cache()` to release all unoccupied cached memory 
        currently held by the CUDA memory allocator, making that memory available for other GPU applications.
        It also calls Python's garbage collector to free up any unreferenced memory in the system.

        Note:
            This method should be used with caution as it can affect the performance of your application 
            by causing memory fragmentation.
        """
        with torch.no_grad():
            torch.cuda.empty_cache()

        gc.collect()

    def get_tokenizer(self):
        """
        Initializes and returns the tokenizer for the specified model card.

        This method loads a tokenizer using the `AutoTokenizer` class from the 
        Hugging Face Transformers library, based on the model card specified 
        in the `self.model_card` attribute. The loaded tokenizer is then 
        assigned to the `self.tokenizer` attribute and returned.

        Returns:
            PreTrainedTokenizer: The tokenizer initialized from the specified model card.
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_card)
        self.tokenizer = tokenizer
        return tokenizer

    def get_dataset(self, dataframe: pd.DataFrame):
        """
        Converts a pandas DataFrame into a Dataset object.

        Args:
            dataframe (pd.DataFrame): The pandas DataFrame to be converted.

        Returns:
            Dataset: A Dataset object created from the given DataFrame.
        """
        return Dataset.from_pandas(dataframe)
    
    def preprocess_text(self, texts: pd.DataFrame, column: str = 'tweet'):
        """
        Preprocesses the text data by tokenizing the specified column in the DataFrame.

        Args:
            texts (pd.DataFrame): The DataFrame containing the text data to be processed.
            column (str, optional): The name of the column in the DataFrame that contains the text data. Defaults to 'tweet'.

        Returns:
            List[Dict[str, Union[List[int], List[str]]]]: The tokenized representation of the text data.
        """
        return self.tokenizer(texts[column]) # TODO: truncate to max len based on average len?

    def show_encoded_text(self, data, idx: int):
        """
        Displays the original, encoded, and decoded text for a given index in the dataset.

        Args:
            data (dict): A dictionary containing the dataset with keys 'tweet' and 'input_ids'.
            idx (int): The index of the tweet to display.

        Prints:
            str: The original tweet text.
            str: The encoded tweet text as input IDs.
            str: The decoded tweet text from the input IDs.
        """
        print(f"Original: {data['tweet'][idx]}")
        print(f"Encoded: {data['input_ids'][idx]}")
        print(f"Decoded: {self.tokenizer.decode(data['input_ids'][idx])}")

    def get_model(self, num_labels: int, id2label: dict, label2id: dict):
        """
        Loads a pre-trained model for sequence classification.

        Args:
            num_labels (int): The number of labels for classification.
            id2label (dict): A dictionary mapping label IDs to label names.
            label2id (dict): A dictionary mapping label names to label IDs.

        Returns:
            AutoModelForSequenceClassification: The pre-trained model configured for sequence classification.
        """
        return AutoModelForSequenceClassification.from_pretrained(self.model_card, num_labels=num_labels, id2label=id2label, label2id=label2id)

    def get_data_collator(self):
        """
        Creates and returns a data collator that dynamically pads the inputs to the maximum length of a batch.

        Returns:
            DataCollatorWithPadding: An instance of DataCollatorWithPadding initialized with the tokenizer.
        """
        return DataCollatorWithPadding(tokenizer=self.tokenizer)

    def compute_metrics(self, output_info):
        """
        Compute various evaluation metrics for model predictions.

        Args:
            output_info (tuple): A tuple containing the model predictions and the true labels.
                - predictions (np.ndarray): The predicted labels from the model.
                - labels (np.ndarray): The true labels.

        Returns:
            dict: A dictionary containing the computed metrics:
                - 'f1': The F1 score (macro average).
                - 'accuracy': The accuracy score.
                - 'precision': The precision score (macro average).
                - 'recall': The recall score (macro average).
        """
        acc_metric = evaluate.load('accuracy')
        f1_metric = evaluate.load('f1')
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")

        predictions, labels = output_info
        predictions = np.argmax(predictions, axis=-1)

        f1 = f1_metric.compute(predictions=predictions, references=labels, average='macro')
        acc = acc_metric.compute(predictions=predictions, references=labels)
        precision = precision_metric.compute(predictions=predictions, references=labels, average='macro')
        recall = recall_metric.compute(predictions=predictions, references=labels, average='macro')

        return {**f1, **acc, **precision, **recall}

    def get_training_args(self, *args, **kwargs):
        """
        Generates and returns the training arguments for the model.

        This method accepts any number of positional and keyword arguments and 
        passes them to the `TrainingArguments` constructor from the HuggingFace 
        Transformers library.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            TrainingArguments: An instance of the TrainingArguments class 
            initialized with the provided arguments.
        """
        return TrainingArguments(*args, **kwargs)

    def get_trainer(self, model, training_args, train_dataset, eval_dataset, data_collator):
        """
        Creates and returns a CustomTrainer instance for training and evaluation.

        Args:
            model (PreTrainedModel): The model to be trained.
            training_args (TrainingArguments): The arguments for training configuration.
            train_dataset (Dataset): The dataset to be used for training.
            eval_dataset (Dataset): The dataset to be used for evaluation.
            data_collator (DataCollator): The data collator to be used for batching.

        Returns:
            CustomTrainer: An instance of CustomTrainer configured with the provided arguments.
        """
        return CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
            class_weights=self.class_weights,
            device=self.device,
        )

    def get_predictions(self, trainer, data):
        """
        Generate predictions and labels from the given trainer and data.

        Args:
            trainer (Trainer): The trainer object used for making predictions.
            data (Dataset): The dataset on which predictions are to be made.

        Returns:
            tuple: A tuple containing two elements:
                - predictions (ndarray): The predicted values.
                - labels (ndarray): The true labels corresponding to the predictions.
        """
        predictions_info = trainer.predict(data)
        predictions, labels = predictions_info.predictions, predictions_info.label_ids

        return predictions, labels
    
    def get_metrics(self, predictions, labels):
        """
        Calculate and return the evaluation metrics based on the given predictions and labels.

        Args:
            predictions (list or array-like): The predicted values.
            labels (list or array-like): The true values.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        return self.compute_metrics([predictions, labels])

    def get_confusion_matrix(self, predictions, labels):
        """
        Generates and displays a confusion matrix for the given predictions and labels.

        Args:
            predictions (numpy.ndarray): The predicted probabilities or logits for each class.
            labels (numpy.ndarray): The true labels for the data.

        Returns:
            None: This function displays the confusion matrix plot and does not return any value.
        """
        cm = confusion_matrix(y_true=labels, y_pred=np.argmax(predictions, axis=-1))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-sexist', 'Sexist'])
        disp.plot(cmap='viridis')
        plt.show()

    def get_precision_recall(self, predictions, labels):
        """
        Calculate and display the precision-recall curve for the given predictions and labels.

        Args:
            predictions (np.ndarray): The predicted probabilities or logits from the model.
            labels (np.ndarray): The true labels for the data.

        Returns:
            None: This function displays the precision-recall curve using matplotlib.
        """
        display = PrecisionRecallDisplay.from_predictions(y_true=labels, y_pred=np.argmax(predictions, axis=-1), name='Cardiff Twitter RoBERTa Base', plot_chance_level=True)
        _ = display.ax_.set_title("2-class Precision-Recall curve")

    def contains_OOV(self, text):
        """
        Return the out-of-vocabulary (OOV) words in a given text.
        """
        vocab_words = list(self.tokenizer.vocab.keys())
        return set(word for word in text.split() if word not in vocab_words)

    def review_errors(self, original_data, preprocessed_data, predictions, labels, verbose: bool = False):
        """
        Analyzes and reviews errors in model predictions.

        Args:
            original_data (pd.DataFrame): The original dataset containing the true labels.
            preprocessed_data (pd.DataFrame): The preprocessed dataset used for predictions.
            predictions (np.ndarray): The model's predictions.
            labels (np.ndarray): The true labels.
            verbose (bool, optional): If True, prints detailed information about each error. Defaults to False.

        Returns:
            list: A list of tweet IDs where the model made incorrect predictions.

        Prints:
            - The total number of errors found.
            - Detailed information about each error if verbose is True.
            - The number and percentage of errors due to Out-Of-Vocabulary (OOV) words.
        """
        errors = []
        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if np.argmax(pred) != label:
                errors.append(i)

        print(f"Found {len(errors)} errors out of {len(labels)} samples.")

        errors_due_to_OOV = 0
        wrong_tweet_ids = []
        for i in errors:
            tweet_text = preprocessed_data['tweet'][i]
            tweet_id = preprocessed_data['id_EXIST'][i]
            wrong_tweet_ids.append(tweet_id)
            oov_found = self.contains_OOV(tweet_text)
            original_labels = original_data[original_data['id_EXIST'] == tweet_id]['labels_task1'].values[0]
            if oov_found:
                errors_due_to_OOV += 1
            if verbose:
                print(f"""
                Contains OOV: {oov_found if len(oov_found) > 0 else 'No'}
                Tweet: {tweet_text}
                Predicted: {predictions[i]}
                True Label: {labels[i]}
                Raw Labels: {original_labels}

                """)
        print(f"Errors due to OOV: {errors_due_to_OOV}, {errors_due_to_OOV / len(errors) * 100:.2f}%")
        return wrong_tweet_ids

# Since we know that there is a slightly imbalance in the dataset labels, instead of using the default Trainer class of HuggingFace, we can write a custom class inheriting from the Trainer class and override the `compute_loss` method to compute the loss with the class weights computed from the dataset labels. In this way the loss function will take into account the imbalance in the dataset labels and will give more importance to the minority class. The impact of this change is that the F1 and accuracy scores will be more similar to each other since the model will be trained to give more importance to the minority class w.r.t. the majority class.

# Taken from https://github.com/huggingface/transformers/blob/main/src/transformers/trainer.py#L3700 (with some minor changes removing useless parts)
class CustomTrainer(Trainer):
    def __init__(self, class_weights, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # You pass the class weights when instantiating the Trainer
        self.class_weights = class_weights
        self.device = device

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.

            # Do not
            # loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            logits = outputs.get('logits')
            criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(self.device))
            loss = criterion(logits, inputs['labels'])

        return (loss, outputs) if return_outputs else loss