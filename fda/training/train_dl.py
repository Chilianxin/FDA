from __future__ import annotations

import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    args = ap.parse_args()
    print('Stage A placeholder: pretrain DL encoders (ranking + quantile)')
    for e in range(args.epochs):
        print(f'Epoch {e+1}: training...')


if __name__ == '__main__':
    main()

