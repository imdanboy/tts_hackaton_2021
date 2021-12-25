#!/usr/bin/env python3


import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("emotion", type=str)
    parser.add_argument("utt2eid", type=str)
    args = parser.parse_args()

    
    lines = Path(args.emotion).open('r').readlines()








if __name__ == "__main__":
    main()
