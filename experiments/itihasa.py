"""
Contains the preprocessing pipeline for the itihasa dataset.


Note: Much of this logic can be reused for other datasets
(e.g. vocabulary generation, tokenization, etc.); feel free to split
into multiple files when that happens. Things are only kept centralized
here due to development time constraints.
"""
import datasets
import tokenizers
import torch
import os

# Create a cache folder, used to speed up expensive operations between runs
from pathlib import Path
CACHE_FOLDER = "./cache"
Path(CACHE_FOLDER).mkdir(parents=True, exist_ok=True)

def train_itihasa_tokenizers(merged_data):
    """
    Trains BPE tokenizers over the full Itihasa dataset
    
    Args:
        merged_data: datasets.Dataset       Contains a concatenation of all
                                            of Itihasa's dataset splits, since
                                            tokenizer should be trained across
                                            the full language.
    Return:
        tokenizers: Tuple       A tuple with the following fields:
            en_tokenizer        A tokenizer trained on the English dataset
            sn_tokenizer        A tokenizer trained on the Sanskrit dataset
            
    """

    # Using Byte-Pair Encoding for tokenization
    en_bpe = tokenizers.Tokenizer(tokenizers.models.BPE())
    sn_bpe = tokenizers.Tokenizer(tokenizers.models.BPE())

    en_bpe_cache_file, sn_bpe_cache_file = CACHE_FOLDER + "/en_bpe", CACHE_FOLDER + "/sn_bpe"
    if os.path.isfile(en_bpe_cache_file) and os.path.isfile(sn_bpe_cache_file):
        return (en_bpe.from_file(en_bpe_cache_file), sn_bpe.from_file(sn_bpe_cache_file))

    # Use whitespace as a word delimiter
    en_bpe.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    sn_bpe.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

    english_corpus_iter =  map(lambda x: x['en'], merged_data['translation'])
    sanskrit_corpus_iter =  map(lambda x: x['sn'], merged_data['translation'])

    corpus_length = merged_data.num_rows

    trainer = tokenizers.trainers.BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    en_bpe.train_from_iterator(english_corpus_iter, length=corpus_length, trainer=trainer)
    sn_bpe.train_from_iterator(sanskrit_corpus_iter, length=corpus_length)

    en_bpe.save(CACHE_FOLDER + "/en_bpe")
    sn_bpe.save(CACHE_FOLDER + "/sn_bpe")

    return (en_bpe, sn_bpe)


def load_itihasa():
    """
    Loads the itihasa dataset.

    Returns:
        training_data: Dataset       Contains Itihasa's training split
        validation_data: Dataset     Contains Itihasa's validation split
        test_data: Dataset           Contains Itihasa's test split
    """
    dataset = datasets.load_dataset("rahular/itihasa")

    training_data = dataset['train']
    validation_data = dataset['validation']
    test_data = dataset['test']
    return (training_data, validation_data, test_data)


def preprocess_parallel_pair(
    raw_english_sentence, raw_sanskrit_sentence, tokenizers):
    """
    Preprocesses a single Sanskrit and English parallel pair

    Args:
        raw_english_sentence        A sentence in English
        raw_sanskrit_sentence       The corresponding sentence in Sanskrit
        tokenizers: Tuple       A tuple with the following fields:
            en_tokenizer        A tokenizer trained on the English dataset
            sn_tokenizer        A tokenizer trained on the Sanskrit dataset

    Returns:
        preprocessed_text: Tuple    A Tuple with the following fields:
            preprocessed_english    Preprocessed English text
            preprocessed_sanskrit   Preprocessed Sanskrit text
    """
    # Tokenize sentence
    en_tokenizer, sn_tokenizer = tokenizers
    tokenized_en = en_tokenizer.encode(raw_english_sentence)
    tokenized_sn = sn_tokenizer.encode(raw_sanskrit_sentence)

    return (torch.LongTensor(tokenized_en.ids),
            torch.LongTensor(tokenized_sn.ids))


class ItihasaDataset(torch.utils.data.Dataset):
    """
    An ItihasaDataset houses the preprocessed Itihasa dataset (see
    https://github.com/rahular/itihasa), ready for model consumption.
    """
    def __init__(self, parallel_text, tokenizer):
        """
        Args:
            parallel_text   The datasets.Dataset holding the parallel text
            tokenizers      The tokenizers used to tokenize each sentence. Note
                            that it should already be trained. Tuple with the
                            following fields:
                en_tokenizer        English tokenizer
                sn_tokenizer        Sanskrit tokenizer
        """
        self._parallel_text = parallel_text
        self._tokenizer = tokenizer

    def __getitem__(self, idx):
        """
        Returns a preprocessed pair of parallel Sanskrit-->English text

        Args:
            idx         Index of the dataset to access

        Return:
            preprocessed_parallel_pair: Tuple
                source      Preprocessed source sentence (Sanskrit sentence
                            converted to list of tokens)
                target      Preprocessed source sentence (English sentence
                            converted to list of tokens)
        """
        parallel_pair = self._parallel_text['translation'][idx]
        en_sentence, sn_sentence = parallel_pair['en'], parallel_pair['sn']
        
        preprocessed_english, preprocessed_sanskrit = preprocess_parallel_pair(
            en_sentence, sn_sentence, self._tokenizer)

        source, target = preprocessed_sanskrit, preprocessed_english
        return (source, target)


    def __len__(self):
        return self._parallel_text.num_rows

if __name__ == "__main__":
    """
    Example Usage
    """
    # Download the Itihasa dataset
    training_data, validation_data, test_data = load_itihasa()

    # Combine all data splits for tokenizer training
    merged_data = datasets.concatenate_datasets(
        (training_data, validation_data, test_data))

    # Train the tokenizers on the existing corpora
    tokenizers = train_itihasa_tokenizers(merged_data)
    
    # Create Torch Datasets for each split of the Itihasa dataset
    # (useful to get training up-and-running)
    itihasa_dataset_train = ItihasaDataset(training_data, tokenizers)
    itihasa_dataset_val = ItihasaDataset(validation_data, tokenizers)
    itihasa_dataset_test = ItihasaDataset(test_data, tokenizers)

    # Create a torch Dataset from the full Itihasa dataset (useful
    # if you want a custom train/val/test split)
    itihasa_dataset_full = ItihasaDataset(merged_data, tokenizers)
