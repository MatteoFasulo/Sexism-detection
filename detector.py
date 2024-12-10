import gc
import os
import requests
from pathlib import Path
import re
import json
from typing import OrderedDict
import copy


import unicodedata

import numpy as np
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score

from tqdm import tqdm
import matplotlib.pyplot as plt

import nltk
from nltk.stem import WordNetLemmatizer

import gensim
import gensim.downloader as gloader

import torch

from datasets import Dataset
from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import Trainer
import evaluate

class SexismDetector:
    def __init__(self):
        URL_PATTERN_STR = r"""(?i)((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info
                      |int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|
                      bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|
                      cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|
                      gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|
                      la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|
                      nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|
                      sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|
                      uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]
                      *?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’])|(?:(?<!@)
                      [a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name
                      |post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn
                      |bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg
                      |eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id
                      |ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|
                      md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|
                      ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|
                      sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|
                      za|zm|zw)\b/?(?!@)))"""
        self.URL_PATTERN = re.compile(URL_PATTERN_STR, re.IGNORECASE)
        self.HASHTAG_PATTERN = re.compile(r'#\w*')
        self.MENTION_PATTERN = re.compile(r'@\w*')
        self.EMOJIS_PATTERN = re.compile(u'([\U00002600-\U000027BF])|([\U0001f300-\U0001f64F])|([\U0001f680-\U0001f6FF])')
        self.AND_PATTERN = re.compile(r'&amp;')
        self.PUNCT_REPEAT_PATTERN = re.compile(r'([!?.]){2,}')
        self.ELONG_PATTERN = re.compile(r'\b(\S*?)(.)\2{2,}\b')
        self.WORD_PATTERN = re.compile(r'[^\w<>\s]')
        self.SEED = 1337
        self.DATA_FOLDER = Path('data')
        self.MODEL_FOLDER = Path('models')
        self.columns_to_maintain = ['id_EXIST', 'lang', 'tweet', 'hard_label_task1']
        self.UNK_TOKEN = '[UNK]'
        self.PAD_TOKEN = '[PAD]'

    def download_corpus(self, url: str, filename: str) -> None:
        """
        Downloads a text corpus from a given URL and saves it to a specified filename within the data folder.

        Args:
            url (str): The URL from which to download the corpus.
            filename (str): The name of the file to save the downloaded corpus.

        Raises:
            requests.exceptions.HTTPError: If the HTTP request returned an unsuccessful status code.

        Side Effects:
            Creates the data folder if it does not exist.
            Writes the downloaded corpus to the specified file.
        """
        if not self.DATA_FOLDER.exists():
            self.DATA_FOLDER.mkdir(parents=True)
            print(f"Created folder {self.DATA_FOLDER}.")

        response = requests.get(url)
        response.raise_for_status()
        with open(self.DATA_FOLDER / filename, 'w', encoding='utf-8') as f:
            f.write(response.text)

    def load_corpus(self, filename: str, *args, **kwargs) -> pd.DataFrame:
        """
        Load a corpus from a JSON file.

        Parameters:
        filename (str): The name of the JSON file to load.
        *args: Variable length argument list to pass to pandas read_json.
        **kwargs: Arbitrary keyword arguments to pass to pandas read_json.

        Returns:
        DataFrame: A pandas DataFrame containing the loaded corpus.
        """
        return pd.read_json(self.DATA_FOLDER / filename, *args, **kwargs)

    @staticmethod
    def majority_voting(votes: list[str]) -> str:
        """
        Determines the majority vote from a list of votes.
        Args:
            votes (list[str]): A list of votes, where each vote is either "YES" or "NO".
        Returns:
            str: The result of the majority vote. Returns "YES" if there are more "YES" votes,
                "NO" if there are more "NO" votes, and "NEUTRAL" in case of a tie.
        """
        total_num_votes = len(votes)
        yes_votes = votes.count("YES")
        no_votes = total_num_votes - yes_votes

        if yes_votes > no_votes:
            return "YES"
        elif no_votes > yes_votes:
            return "NO"
        else:
            return "NEUTRAL" # This will be the case when there is a tie (removed later)

    def preprocess_text(self, text: str) -> str:
        """
        Preprocesses the input text by removing or replacing specific patterns.

        Args:
            text (str): The input text to be preprocessed.

        Returns:
            str: The preprocessed text with URLs, mentions, hashtags, emojis,
                 special characters removed, 'and' replaced, and extra spaces trimmed.
        """
        # Convert URL to <URL> so that GloVe will have a vector for it
        text = re.sub(self.URL_PATTERN, ' <URL>', text)
        # Add spaces around slashes
        text = re.sub(r"/", " / ", text)
        # Replace mentions with <USER>
        text = re.sub(self.MENTION_PATTERN, ' <USER> ', text)
        # Replace numbers with <NUMBER>
        text = re.sub(r"[-+]?[.\d]*[\d]+[:,.\d]*", " <NUMBER> ", text)
        # Replace hashtags with <HASHTAG>
        text = re.sub(self.HASHTAG_PATTERN, ' <HASHTAG> ', text)
        #text = self.AND_PATTERN.sub('and', text) # &amp; already in the Vocab of GloVe-twitter
        # Replace multiple punctuation marks with <REPEAT>
        text = re.sub(self.PUNCT_REPEAT_PATTERN, lambda match: f" {match.group(1)} <REPEAT> ", text)
        # Replace elongated words with <ELONG>
        text = re.sub(self.ELONG_PATTERN, lambda match: f" {match.group(1)}{match.group(2)} <ELONG> ", text)
        #text = emoji.replace_emoji(text, replace='') # some emojis are in the vocab so we do not remove them, the others will be OOVs
        text = text.strip()
        # Get only words
        text = re.sub(self.WORD_PATTERN, ' ', text)
        text = text.strip()
        # Convert stylized Unicode characters to plain text (removes bold text, etc.)
        text = ''.join(c for c in unicodedata.normalize('NFKD', text) if not unicodedata.combining(c))
        return text

    def lemmatize_text(self, text: str) -> str:
        """
        Lemmatizes the input text using the WordNet lemmatizer.

        This method attempts to lemmatize each word in the input text. If the WordNet
        data is not available, it will download the necessary data and retry.

        Args:
            text (str): The input text to be lemmatized.

        Returns:
            str: The lemmatized text.
        """
        lemmatizer = WordNetLemmatizer()
        downloaded = False
        while not downloaded:
            try:
                lemmatizer.lemmatize(text)
                downloaded = True
            except LookupError:
                print("Downloading WordNet...")
                nltk.download('wordnet')
        return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

    @staticmethod
    def text_diff(original_text: str, preprocessed_text: str, random: bool = True) -> None:
        """
        Displays a comparison between an original tweet and its preprocessed version.
        Args:
            original_text (str): The original text data containing tweets.
            preprocessed_text (str): The preprocessed text data containing tweets.
            random (bool, optional): If True, a random tweet is selected for comparison.
                                     If False, the first tweet is selected. Defaults to True.
        Returns:
            None
        """
        if random:
            idx = np.random.randint(0, preprocessed_text.shape[0])
        else:
            idx = 0

        print(f"Original tweet:\n{original_text['tweet'].iloc[idx]}")
        print(f"Processed tweet:\n{preprocessed_text['tweet'].iloc[idx]}")

    def load_glove(self, model_name: str = 'glove-wiki-gigaword', embedding_dim: int = 50) -> gensim.models.keyedvectors.KeyedVectors:
        """
        Loads the GloVe model with the specified name and embedding dimension.
        Args:
            model_name (str): The name of the GloVe model to load. Default is 'glove-wiki-gigaword'.
            embedding_dim (int): The dimension of the word embeddings. Default is 50.
        Returns:
            gensim.models.keyedvectors.KeyedVectors: The loaded GloVe model.
        Raises:
            Exception: If there is an error in downloading or loading the model.
        Notes:
            - If the model folder does not exist, it will be created.
            - If the model is not already downloaded, it will be downloaded and saved to the specified path.
            - If the model is already downloaded, it will be loaded from the specified path.
        """
        self.EMBEDDING_DIM = embedding_dim

        if not self.MODEL_FOLDER.exists():
            self.MODEL_FOLDER.mkdir(parents=True)
            print(f"Created folder {self.MODEL_FOLDER}.")

        model_path = self.MODEL_FOLDER / f"{model_name}-{embedding_dim}"
        if not model_path.exists():
            print(f"Downloading {model_name} model...")
            glove_model = gloader.load(f"{model_name}-{embedding_dim}")
            print(f"Model downloaded! Saving to {model_path}")
            glove_model.save(str(model_path))
            print(f"Model saved to {self.MODEL_FOLDER / f'{model_name}-{embedding_dim}'}")
        else:
            glove_model = gensim.models.keyedvectors.KeyedVectors.load(str(model_path))
        return glove_model

    def get_vocab(self, data: pd.DataFrame) -> tuple[OrderedDict, OrderedDict]:
        """
        Generates vocabulary mappings from a given dataset.
        Args:
            data (pd.DataFrame): A pandas DataFrame containing the dataset with a column 'tweet'.
            word_listing (list, optional): A list of words to include in the vocabulary. If None, the vocabulary
                                           will be built from the dataset. Defaults to None.
        Returns:
            tuple[OrderedDict, OrderedDict]: A tuple containing two OrderedDicts:
                - idx_to_word: Mapping from index to word.
                - word_to_idx: Mapping from word to index.
        """
        idx_to_word = OrderedDict()
        word_to_idx = OrderedDict()

        curr_idx = 0
        for sentence in data.tweet.values:
            tokens = sentence.split()
            for token in tokens:
                if token not in word_to_idx:
                    word_to_idx[token] = curr_idx
                    idx_to_word[curr_idx] = token
                    curr_idx += 1

        word_listing = list(idx_to_word.values())
        return idx_to_word, word_to_idx, word_listing

    def co_occurrence_count(self, df: pd.DataFrame, idx_to_word, word_to_idx, window_size: int = 10) -> np.ndarray:

        vocab_size = len(idx_to_word)
        co_occurrence_matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

        for sentence in tqdm(df.tweet.values):
            tokens = sentence.split()
            for pos, token in enumerate(tokens):
                start = max(0, pos - window_size)
                end = min(pos + window_size + 1, len(tokens))

                first_word_index = word_to_idx[token]

                for pos2 in range(start, end):
                    if pos2 != pos:
                        second_token = tokens[pos2]
                        second_word_index = word_to_idx[second_token]
                        co_occurrence_matrix[first_word_index, second_word_index] += 1

        return co_occurrence_matrix

    def co_occurrence_dataframe(self, co_occurrence_matrix: np.ndarray, idx_to_word: OrderedDict) -> pd.DataFrame:
        """
        Converts a co-occurrence matrix into a pandas DataFrame with word indices as row and column labels.
        Args:
            co_occurrence_matrix (np.ndarray): A co-occurrence matrix with word indices as row and column indices.
            idx_to_word (OrderedDict): A mapping from word index to word.
        Returns:
            pd.DataFrame: A pandas DataFrame with word indices as row and column labels.
        """
        return pd.DataFrame(co_occurrence_matrix, index=idx_to_word.values(), columns=idx_to_word.values())

    def get_augmented_vocab(self, emb_model: gensim.models.keyedvectors.KeyedVectors, train_words: list, co_occurrence_df: pd.DataFrame, save: bool = False) -> gensim.models.keyedvectors.KeyedVectors:
        """
        Augments the given embedding model with new tokens from the training words list. If a token is not found in the
        embedding model, a random vector is generated for it. Optionally saves the updated vocabulary to a JSON file.
        Args:
            emb_model (gensim.models.keyedvectors.KeyedVectors): The embedding model to augment.
            train_words (list): A list of words to add to the embedding model.
            co_occurrence_df (pd.DataFrame): A DataFrame containing co-occurrence data for the training words.
            save (bool, optional): If True, saves the updated vocabulary to a JSON file. Defaults to False.
        Returns:
            gensim.models.keyedvectors.KeyedVectors: The augmented embedding model.
        """
        new_tokens = []
        new_vectors = []

        for token in train_words:
            try:
                embedding_vec = emb_model.get_vector(token)
            except KeyError:
                try:
                    # Handle missing token in embedding model
                    top_5_most_frequent = co_occurrence_df.loc[token].sort_values(ascending=False).index.tolist()[:5]

                    # Filter words that are in the embedding model
                    valid_words = [word for word in top_5_most_frequent if word in emb_model.key_to_index]

                    if valid_words: # TODO: check that valid words as atleast 2 words otherwise mean won't be significant
                        # Calculate the average vector for valid words
                        embedding_vec = np.mean([emb_model.get_vector(word) for word in valid_words], axis=0)
                    else:
                        # Handle the case where no valid co-occurring words exist
                        raise ValueError(f"No valid co-occurring words found for token: {token}")

                except Exception as e:
                    print(f"Generating random vector for token '{token}': {e}")
                    embedding_vec = np.random.uniform(low=-0.05, high=0.05, size=self.EMBEDDING_DIM)

            new_tokens.append(token)
            new_vectors.append(embedding_vec)

        emb_model.add_vectors(new_tokens, new_vectors)
        # add the UNK token to the embedding model with the vector which is the average of all the vectors
        emb_model.add_vectors(["[UNK]", "[PAD]"], [np.mean(emb_model.vectors, axis=0), np.zeros(self.EMBEDDING_DIM)])

        if save:
            vocab_path = self.DATA_FOLDER / 'vocab.json'
            print(f"Saving vocab to {vocab_path}")
            with vocab_path.open('w', encoding='utf-8') as f:
                json.dump(emb_model.key_to_index, f, indent=4)
            print("Vocab saved!")

        return emb_model

    def get_oov(self, embedding_model: gensim.models.keyedvectors.KeyedVectors, word_listing: list) -> set:
        """
        Returns a list of out-of-vocabulary (OOV) words from a given list of words.
        Args:
            embedding_model (gensim.models.keyedvectors.KeyedVectors): The word embedding model containing known words.
            word_listing (list): A list of words to check against the embedding model.
        Returns:
            set: A set of out-of-vocabulary words.
        """
        return set(word_listing).difference(set(embedding_model.key_to_index.keys()))

    def get_oov_stats(self, embedding_model: gensim.models.keyedvectors.KeyedVectors, word_listing: list) -> None:
        """
        Calculate and print the number and percentage of out-of-vocabulary (OOV) words.
        Args:
            embedding_model (gensim.models.keyedvectors.KeyedVectors): The word embedding model containing known words.
            word_listing (list): A list of words to check against the embedding model.
        Returns:
            None
        """
        OOV_token = self.get_oov(embedding_model, word_listing)
        OOV_percentage = float(len(OOV_token)) * 100 / len(word_listing)

        print(f"Total OOV terms: {len(OOV_token)} ({OOV_percentage:.2f}%)")

    def get_padded_sequences(self, data: pd.Series, embedding_model: gensim.models.keyedvectors.KeyedVectors) -> torch.Tensor:
        """
        Converts a pandas Series of text data into padded sequences of word indices using a given embedding model.

        Args:
            data (pd.Series): A pandas Series containing text data.
            embedding_model (gensim.models.keyedvectors.KeyedVectors): A pre-trained word embedding model.

        Returns:
            torch.Tensor: A tensor containing padded sequences of word indices.
        """
        tokenizer = nltk.tokenize.NLTKWordTokenizer()
        unk_index = 9856

        sequences = [torch.tensor([embedding_model.get_index(word, default=unk_index) for word in tokenizer.tokenize(x)]) for x in data.values]

        return torch.nn.utils.rnn.pad_sequence(sequences, padding_value=embedding_model.get_index(self.PAD_TOKEN), batch_first=True)

    def get_dataloader(self, data: pd.DataFrame, embedding_model: gensim.models.keyedvectors.KeyedVectors, type: str, *args, **kwargs) -> torch.utils.data.DataLoader:
        """
        Creates a DataLoader for the given dataset.

        Args:
            data (pd.DataFrame): DataFrame containing the data. Must have 'tweet' and 'label' columns.
            embedding_model (gensim.models.keyedvectors.KeyedVectors): Pre-trained embedding model to convert tweets to vectors.
            type (str): Type of dataset. Must be one of 'train', 'val', or 'test'.
            *args: Variable length argument list for DataLoader.
            **kwargs: Arbitrary keyword arguments for DataLoader.

        Returns:
            torch.utils.data.DataLoader: DataLoader for the given dataset.

        Raises:
            ValueError: If the type is not one of 'train', 'val', or 'test'.
        """
        if type not in ['train', 'val', 'test']:
            raise ValueError("Invalid type. Must be one of 'train', 'val', or 'test'.")
        padded_sequences = self.get_padded_sequences(data.tweet, embedding_model)
        labels = torch.tensor(data.label.values)
        dataset = TextDataset(padded_sequences, labels)
        return torch.utils.data.DataLoader(dataset, *args, **kwargs)

class TextDataset(torch.utils.data.Dataset):
    """
    A custom dataset class for handling text sequences and their corresponding labels.

    Args:
        sequences (torch.Tensor): A tensor containing the text sequences.
        labels (torch.Tensor): A tensor containing the labels for each text sequence.

    Attributes:
        sequences (torch.Tensor): Stores the text sequences.
        labels (torch.Tensor): Stores the labels for each text sequence.

    Methods:
        __len__(): Returns the number of text sequences in the dataset.
        __getitem__(idx): Returns the text sequence and label at the specified index.
    """
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return self.sequences.shape[0]

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

