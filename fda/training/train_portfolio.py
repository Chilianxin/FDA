from __future__ import annotations

import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=1)
    args = ap.parse_args()
    print('Stage B placeholder: differentiable portfolio + execution pretraining')
    for e in range(args.epochs):
        print(f'Epoch {e+1}: training...')


if __name__ == '__main__':
    main()

