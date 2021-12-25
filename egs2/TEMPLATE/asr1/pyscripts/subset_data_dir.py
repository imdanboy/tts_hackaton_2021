#!/usr/bin/env python3

import argparse
from pathlib import Path
import json




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root", type=str, required=True)
    parser.add_argument("--train", type=str, required=True)
    #parser.add_argument("--train_set", type=str, required=True)
    #parser.add_argument("--train_dev", type=str, required=True)
    #parser.add_argument("--eval_set", type=str, required=True)
    args = parser.parse_args()

    #train_set = Path(args.train_set)
    #train_set.mkdir(parents=True, exist_ok=True)
    #train_dev = Path(args.train_dev)
    #train_dev.mkdir(parents=True, exist_ok=True)
    #eval_set = Path(args.eval_set)
    #eval_set.mkdir(parents=True, exist_ok=True)

    db_root = Path(args.db_root)

    train_json = json.load(open(db_root / "tta-train-v0.1.4.json", 'r'))
    valid_json = json.load(open(db_root / "tta-valid-v0.1.4.json", 'r'))
    test_json = json.load(open(db_root / "tta-test-v0.1.4.json", 'r'))

    def get_utt_id_list(x_json):
        utt_id_x = []
        for items in x_json:
            sentences = items['sentences']
            for sentence in sentences:
                voice_piece = sentence['voice_piece']
                filename = voice_piece['filename']
                utt_id = Path(filename).stem
                #utt_id_x.append(utt_id)
                # literature_id + spk_id + sent_idx -> spk_id + literature_id + sent_idx
                lines = utt_id.split('-')
                modified_utt_id = '-'.join(lines[2:3] + lines[:2] + lines[-1:])
                utt_id_x.append(modified_utt_id)
        return utt_id_x

    utt_id_train = get_utt_id_list(train_json)
    utt_id_valid = get_utt_id_list(valid_json)
    utt_id_test = get_utt_id_list(test_json)


    utt_list_train = Path(args.train) / 'utt_list_train'
    with utt_list_train.open('w') as f:
        for utt_id in utt_id_train:
            f.write(f"{utt_id}" + '\n')
    utt_list_valid = Path(args.train) / 'utt_list_valid'
    with utt_list_valid.open('w') as f:
        for utt_id in utt_id_valid:
            f.write(f"{utt_id}" + '\n')
    utt_list_test = Path(args.train) / 'utt_list_test'
    with utt_list_test.open('w') as f:
        for utt_id in utt_id_test:
            f.write(f"{utt_id}" + '\n')


    utt_list_train_valid = Path(args.train) / 'utt_list_train_valid'
    with utt_list_train_valid.open('w') as f:
        for utt_id in utt_id_train + utt_id_valid:
            f.write(f"{utt_id}" + '\n')


if __name__ == '__main__':
    main()
