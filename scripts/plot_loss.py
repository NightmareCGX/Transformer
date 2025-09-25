# scripts/plot_loss.py
import csv
import sys

import matplotlib.pyplot as plt


def main(log_csv):
    steps, tr, va = [], [], []
    with open(log_csv) as f:
        r = csv.DictReader(f)
        for row in r:
            steps.append(int(row["step"]))
            tr.append(float(row["train_loss"]))
            va.append(float(row["val_loss"]))
    plt.figure()
    plt.plot(steps, tr, label="train")
    plt.plot(steps, va, label="val")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.title("W2 TinyShakespeare")
    out = log_csv.replace(".csv", ".png")
    plt.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main(sys.argv[1])
