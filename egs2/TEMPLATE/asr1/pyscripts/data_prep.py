#!/usr/bin/env python3

import argparse
from pathlib import Path
import json
from tqdm import tqdm

from typing import List



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_root", type=str, required=True)
    parser.add_argument("--scp", type=str, required=True)
    parser.add_argument("--utt2spk", type=str, required=True)
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--emotion", type=str, required=True)
    args = parser.parse_args()

    db_root = Path(args.db_root)

    train_json = json.load(open(db_root / "tta-train-v0.1.4.json", 'r'))
    valid_json = json.load(open(db_root / "tta-valid-v0.1.4.json", 'r'))
    test_json = json.load(open(db_root / "tta-test-v0.1.4.json", 'r'))
    all_json: List = train_json + valid_json + test_json

    #wav_list = db_root / "data2"
    #wav_list = wav_list.rglob("*.wav")

    fout_scp = Path(args.scp).open('w')
    fout_utt2spk = Path(args.utt2spk).open('w')
    fout_text = Path(args.text).open('w')
    fout_emotion = Path(args.emotion).open('w')

    # all_json
    # keys: id, voice, reciter, recite_src, 'sentences'
    #  sentences: List[Dict];
    #   keys: origin_text, 'voice_piece', 'styles', votes,
    #    voice_piece: 'filename', 'tr', ptr, duration
    #    styles: 'emotion', style

    parse_val = []
    for items in all_json:
        sentences = items['sentences']
        for sentence in sentences:
            voice_piece = sentence['voice_piece']
            styles = sentence['styles']
            filename = voice_piece['filename']
            tr = voice_piece['tr']
            assert len(styles) == 1, f"{len(styles)}"
            emotion = styles[0]['emotion']

            utt_id = Path(filename).stem
            wav_fpath = db_root / filename
            spk = utt_id.split('-')[-2]
            lines = utt_id.split('-')
            # literature_id + spk_id + sent_idx -> spk_id + literature_id + sent_idx
            modified_utt_id =  '-'.join(lines[2:3] + lines[:2] + lines[-1:])

            fout_scp.write(f"{modified_utt_id} {wav_fpath}" + '\n')
            fout_utt2spk.write(f"{modified_utt_id} {spk}" + '\n')
            fout_text.write(f"{modified_utt_id} {tr}" + '\n')
            fout_emotion.write(f"{modified_utt_id} {emotion}" + '\n')

            #val = (utt_id, wav_fpath, spk, tr, emotion)
            #parse_val.append(val)

    #parse_val.sort(key=lambda x: x[0])
    #for val in parse_val:
    #    utt_id, wav_fpath, spk, tr, emotion = val
    #    fout_scp.write(f"{utt_id} {wav_fpath}" + '\n')
    #    fout_utt2spk.write(f"{utt_id} {spk}" + '\n')
    #    fout_text.write(f"{utt_id} {tr}" + '\n')
    #    fout_emotion.write(f"{utt_id} {emotion}" + '\n')






if __name__ == '__main__':
    main()
