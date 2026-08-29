import pandas as pd
import torch
import json
import pickle

def encode_column(df, column, p2idx):
    return df[column].apply(
        lambda x: [p2idx[p] for p in x.split()]
        if pd.notna(x) else []
    )

paths = ["l2_arctic_dev.csv", "l2_arctic_train.csv", "l2_arctic_test.csv"]

df = [pd.read_csv(csv_path) for csv_path in paths]

train = pd.read_csv('l2_arctic_train.csv')
dev = pd.read_csv('l2_arctic_dev.csv')
test = pd.read_csv('l2_arctic_test.csv')

phonemes = set()
for file in df:
    for column in ["Transcript", "Canonical"]:
        for text in file[column].dropna():
            # Nếu phoneme được phân cách bằng khoảng trắng
            phonemes.update(text.split())

# Sort để vocab có thứ tự cố định
phonemes = sorted(phonemes)

p2idx_new = {"":0,
         '[PAD]':1}
# Tạo mapping phoneme -> ID
p2idx_new.update({
    phoneme: idx+2
    for idx, phoneme in enumerate(phonemes)
})


idx2p_new = {idx:p for p,idx in p2idx_new.items()}
vocab_new = {'p2idx':p2idx_new,
             'idx2p':idx2p_new}
print(p2idx_new)

df_vocab_new = pd.DataFrame(vocab_new)

with open('vocab_new.pkl', 'wb') as f:
    pickle.dump(vocab_new, f)



for df, output in [
    (train, "l2_arctic_train_id_new.csv"),
    (dev, "l2_arctic_dev_id_new.csv"),
    (test, "l2_arctic_test_id_new.csv")
]:
    df["Canonical"] = encode_column(df, "Canonical", p2idx_new)
    df["Transcript"] = encode_column(df, "Transcript", p2idx_new)

    df.to_csv(output, index=False)






with open('vocab.json', 'r') as f:
    vocab = json.load(f)

p2idx_old = {"":0,
         '[PAD]':1}

p2idx_old.update({phoneme : idx+2 for phoneme,idx in vocab.items()})


idx2p_old = {idx:p for p,idx in p2idx_old.items()}
vocab_old = {'p2idx':p2idx_old,
             'idx2p':idx2p_old}

with open('vocab_old.pkl', 'wb') as f:
    pickle.dump(vocab_old, f)

print(p2idx_old)

train = pd.read_csv('train_canonical_error.csv')
dev = pd.read_csv('dev.csv')
test = pd.read_csv('test.csv')

for df, output in [
    (train, "l2_arctic_train_id_old.csv"),
    (dev, "l2_arctic_dev_id_old.csv"),
    (test, "l2_arctic_test_id_old.csv")
]:
    df["Canonical"] = encode_column(df, "Canonical", p2idx_old)
    df["Transcript"] = encode_column(df, "Transcript", p2idx_old)

    df.to_csv(output, index=False)



