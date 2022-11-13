"""
Train baseline using naive transformer from Attention is All You Need
"""
import itihasa
import torch
import datasets

class TransformerBaseline(torch.nn.Module):
    def __init__(self, d_model, src_vocab_size, tgt_vocab_size, training_mode=True):
        """
        Creates a baseline transformer model using pytorch's default implementation.
        See https://docs.google.com/document/d/1_IzaLWNXLWBvKiuK2lZUowBnE4ncke9FyZ2-gWySAX4/edit#heading=h.lsnewvk98yum

        Args:
            d_model             Embedding dimension

            src_vocab_size      Size of the source language vocabulary

            tgt_vocab_size      Size of the target language vocabulary

            training_mode       Set this to True when training the model,
                                and set to False when running inference.
                                You can also use model.training_mode = False
                                to dynamically change to inference mode.

                                When the model is in training mode, teacher
                                forcing will be used.
        """
        super(TransformerBaseline, self).__init__()

        self.transformer_model = torch.nn.Transformer(d_model=d_model)
        self.input_embedding = torch.nn.Linear(src_vocab_size, d_model)     # TODO Different src/tgt embeddings?
        self.output_embedding = torch.nn.Linear(d_model, sn_vocab_size)
        self.softmax = torch.nn.Softmax(-1)

        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size

        self.training_mode = training_mode


    def forward(self, src_sentence, tgt_sentence):
        """
        Makes a prediction. If the model is in training_mode, will make teacher
        forcing predictions (e.g. computing each token using ground truth).

        Otherwise, will perform autoregressive prediction (using last predictions
        for next prediction).

        Args:
            src_sentence            A sequence of tokens with shape (batch_size, seq_len).
                                    Note that each token should fall in the interval
                                    [0, src_vocab_size - 1]. These tokens should be mapped
                                    from a sentence in the source language.

            tgt_sentence            A sequence of tokens with shape (batch_size, seq_len).
                                    Note that each token should fall in the interval
                                    [0, tgt_vocab_size - 1]. These tokens should be mapped
                                    from a sentence in the source language.
                                    Note: In inference mode, this should just be [<BOS>].
        """
        src = torch.nn.functional.one_hot(src_sentence, self.src_vocab_size)
        src_embedding = self.input_embedding(src.float())

        tgt = torch.nn.functional.one_hot(tgt_sentence, self.tgt_vocab_size)
        tgt_embedding = self.input_embedding(tgt.float())    # TODO Target and source need different embeddings?

        tgt_len = tgt.shape[0]

        if self.training_mode:      # Teacher forcing
            # See https://docs.google.com/document/d/1yRyJVZDE_jeYudKEONVvQYJrUI0JvlklRgjQEX6JT7Y/edit?usp=sharing
            mask = self.transformer_model.generate_square_subsequent_mask(tgt_len)

            # Note that out has shape [T, d_model], where T is the target len and
            # d_model is the embedding dimension
            out = self.transformer_model(src_embedding, tgt_embedding, tgt_mask=mask)
            prediction = self.softmax(self.output_embedding(out))
            return prediction
        else:
            # TODO Could use naive decoding or beam search here.
            raise NotImplementedError("Have not implemented autoregressive prediction yet")

def train_model(model, train_dataloader, optimizer, criterion, nepochs):
    for epoch in range(nepochs):
        for sn_train, en_train in train_dataloader:
            # Forward pass
            prediction = model.forward(
                src_sentence=sn_train, tgt_sentence=en_train)

            # Ground truth for loss
            one_hot_tgt = torch.nn.functional.one_hot(en_train, en_vocab_size)

            # Squash along batch size, giving shape [batch_size * timesteps, num_classes]
            # This puts the tensors in an acceptable shape for loss computation.
            squashed_predictions = prediction.view(-1, prediction.shape[-1])
            squashed_ground_truth = one_hot_tgt.view(-1, one_hot_tgt.shape[-1]).float()

            # Take loss, calculate gradient, update weights
            loss = criterion(squashed_predictions, squashed_ground_truth)
            loss.backward()
            optimizer.step()

            # Display data
            # TODO Need tensorboard / the pytorch equivalent
            print(loss)

# TODO Would be nice to have args / a config file containing hyperparams
# That would prevent us from having to edit this file to tune them.
if __name__ == "__main__":
    # Download the Itihasa dataset
    training_data, validation_data, test_data = itihasa.load_itihasa()

    # Combine all data splits for tokenizer training
    merged_data = datasets.concatenate_datasets(
        (training_data, validation_data, test_data))

    # Train the tokenizers on the existing corpora
    tokenizers = itihasa.train_itihasa_tokenizers(merged_data)
    
    # TODO Note - training bottleneck coming from loading samples from the
    # ItihasaDataset. Reducing this load time would result in significant
    # speedup.

    # Create Torch Datasets for each split of the Itihasa dataset
    # (useful to get training up-and-running)
    itihasa_dataset_train = itihasa.ItihasaDataset(training_data, tokenizers)
    itihasa_dataset_val = itihasa.ItihasaDataset(validation_data, tokenizers)
    itihasa_dataset_test = itihasa.ItihasaDataset(test_data, tokenizers)

    # Pads all sentences in a batch. Note that source and target
    # batches are padded separately.
    def pad_parallel_pair(batch_of_parallel_pairs):
        batch_of_source_sentences = [src for src, tgt in batch_of_parallel_pairs]
        batch_of_target_sentences = [tgt for src, tgt in batch_of_parallel_pairs]

        return (torch.nn.utils.rnn.pad_sequence(batch_of_source_sentences),
                torch.nn.utils.rnn.pad_sequence(batch_of_target_sentences))

    # Create dataloaders for batching
    train_dataloader = torch.utils.data.DataLoader(
        itihasa_dataset_train, batch_size=3, shuffle=True,
        collate_fn=pad_parallel_pair, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(
        itihasa_dataset_val, batch_size=64, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(
        itihasa_dataset_test, batch_size=64, shuffle=True)

    # Get tokenizers + vocab size from dataset
    sn_tokenizer, en_tokenizer = tokenizers
    en_vocab_size = en_tokenizer.get_vocab_size()
    sn_vocab_size = sn_tokenizer.get_vocab_size()

    # Instantiate model
    baseline_model = TransformerBaseline(
        d_model=512, src_vocab_size=sn_vocab_size, tgt_vocab_size=en_vocab_size,
        training_mode=True)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=baseline_model.parameters())

    # Perform training
    train_model(baseline_model, train_dataloader, optimizer, criterion, nepochs=20)
