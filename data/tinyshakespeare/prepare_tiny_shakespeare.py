import os

import numpy as np


def main():
    in_path = "/home/ezrain/PycharmProjects/Transformer/data/tinyshakespeare/input.txt"
    out_dir = "/home/ezrain/PycharmProjects/Transformer/data/tinyshakespeare"
    validation_ratio = 0.01

    with open(in_path, "rb") as f:
        data = f.read()

    n = len(data)
    split = int(n * (1 - validation_ratio))
    train_data = data[:split]
    val_data = data[split:]

    train_ids = np.frombuffer(train_data, dtype=np.uint8)
    val_ids = np.frombuffer(val_data, dtype=np.uint8)

    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "train.npy"), train_ids)
    np.save(os.path.join(out_dir, "val.npy"), val_ids)

    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        f.write('{"vocabulary size": 256, "tokenizer": "byte-level"}\n')

    print("Prepared: train.py / val.py / meta.json")


if __name__ == "__main__":
    main()
