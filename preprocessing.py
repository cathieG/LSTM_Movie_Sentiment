import pandas as pd
import re
import json
from collections import Counter

class DataPreprocessor:
    def __init__(self, top_k=10000, seq_len=150): # increased top_k from 500 to 10000 to cover more vocabs
        """
        top_k: number of most frequent words to keep
        seq_len: target sequence length for padding/truncation
        """
        self.top_k = top_k
        self.seq_len = seq_len
        self.tokens2index = None

    # ----------------------------
    # Step 1: Read Data
    # ----------------------------
    def read_data(self, train_path, test_path):
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        return train_df, test_df

    # ----------------------------
    # Step 2: Clean Text
    # ----------------------------
    def clean_text(self, df):
        df['clean_text'] = df['Content'].apply(
            lambda x: re.sub(r'[^\w\s]', ' ', str(x)).lower().strip()
        )
        return df

    # ----------------------------
    # Step 3: Filter Sequence Length
    # ----------------------------
    def filter_by_length(self, df, min_len=100, max_len=600): #min and max length defined by Professor's code
        df['seq_words'] = df['clean_text'].apply(str.split)
        df['seq_len'] = df['seq_words'].apply(len)
        print(df['seq_len'].describe())
        df = df[(df['seq_len'] >= min_len) & (df['seq_len'] <= max_len)]
        return df

    # ----------------------------
    # Step 4: Convert Labels
    # ----------------------------
    def convert_labels(self, df):
        df['Label'] = df['Label'].map({'pos': 1, 'neg': 0})
        return df

    # ----------------------------
    # Step 5: Build Vocabulary
    # ----------------------------
    def build_vocab(self, df):
        all_tokens = [w for words in df['seq_words'] for w in words]
        counts = Counter(all_tokens)
        most_common = counts.most_common(self.top_k)

        tokens2index = {w: i + 2 for i, (w, _) in enumerate(most_common)}
        tokens2index['<pad>'] = 0
        tokens2index['<unk>'] = 1

        self.tokens2index = tokens2index

        with open('data/tokens2index.json', 'w') as f:
            json.dump(tokens2index, f, indent=4)
        print(f"Saved vocab: {len(tokens2index)} tokens")

    # ----------------------------
    # Step 6: Encode Sequences
    # ----------------------------
    def encode_sequences(self, df):
        df['input_x'] = df['seq_words'].apply(
            lambda words: [self.tokens2index.get(w, 1) for w in words]
        )
        return df

    # ----------------------------
    # Step 7: Pad / Truncate
    # ----------------------------
    def pad_truncate(self, df):
        def fix_length(seq):
            if len(seq) >= self.seq_len:
                return seq[:self.seq_len]
            else:
                return seq + [0] * (self.seq_len - len(seq))
        df['input_x'] = df['input_x'].apply(fix_length)
        return df

    # ----------------------------
    # Full pipeline
    # ----------------------------
    def process(self, train_path, test_path):
        train_df, test_df = self.read_data(train_path, test_path)

        # Clean and prepare text
        for df in [train_df, test_df]:
            self.clean_text(df)

        # Filter length
        train_df = self.filter_by_length(train_df)
        test_df = self.filter_by_length(test_df)

        # Convert labels
        train_df = self.convert_labels(train_df)
        test_df = self.convert_labels(test_df)

        # Build vocabulary from training set
        self.build_vocab(train_df)

        # Encode and pad
        for df in [train_df, test_df]:
            self.encode_sequences(df)
            self.pad_truncate(df)

        # Save outputs
        train_df.to_csv('data/training_data.csv', index=False)
        test_df.to_csv('data/test_data.csv', index=False)
        print("Saved preprocessed train/test CSVs.")



if __name__ == "__main__":
    preprocessor = DataPreprocessor(top_k=10000, seq_len=150) # parameters copied from the recommendations
    preprocessor.process(
        train_path='data/training_raw_data.csv',
        test_path='data/test_raw_data.csv'
    )
