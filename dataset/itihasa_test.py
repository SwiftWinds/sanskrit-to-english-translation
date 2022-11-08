
from itihasa import ItihasaDataset, load_itihasa, train_itihasa_tokenizers
from fuzzywuzzy import fuzz

def test_tokenization():
    """
    Ensure that the Itihasa tokenization does not destroy information
    """
    training_data, validation_data, test_data = load_itihasa()
    tokenizers = train_itihasa_tokenizers(training_data)
    itihasa_dataset_train = ItihasaDataset(training_data, tokenizers)

    en_tokenizer, sn_tokenizer = tokenizers

    idx = 42        # Take a random datapoint
    en_ground_truth = training_data[idx]['translation']['en']
    sn_ground_truth = training_data[idx]['translation']['sn']
    sn_tokens, en_tokens = itihasa_dataset_train.__getitem__(idx)
    en_reconstructed = en_tokenizer.decode(en_tokens)
    sn_reconstructed = sn_tokenizer.decode(sn_tokens)

    # Ensure that the string reconstructed from the token IDs 
    # matches the original input. Replacing all spaces with ""
    # basically increases the ratio, making it easier to verify
    # correctness (without changing the reliability of this test).
    en_reconstruction_ratio = fuzz.partial_ratio(
        en_ground_truth.replace(" ", ""), en_reconstructed.replace(" ", ""))

    sn_reconstruction_ratio = fuzz.partial_ratio(
        sn_ground_truth.replace(" ", ""), sn_reconstructed.replace(" ", ""))

    assert en_reconstruction_ratio > 90
    assert sn_reconstruction_ratio > 90

if __name__ == "__main__":
    """
    For debugging tests
    """
    test_tokenization()
