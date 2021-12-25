from pathlib import Path
import argparse
import yaml

import numpy as np
import torch

from espnet2.tasks.tts import TTSTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter

import json
from scipy.io.wavfile import write
from espnet2.hifi_gan.models import Generator
from espnet2.hifi_gan.meldataset import MAX_WAV_VALUE
from espnet2.hifi_gan.env import AttrDict




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_list", type=str, default="token_list/phn_g2pk/tokens.txt")
    parser.add_argument("--g2p_type", type=str, default="g2pk")
    parser.add_argument("--output_dir", type=str, default="wavs")
    parser.add_argument("--ngpu", type=int, default=1)
    parser.add_argument("--speed_control_alpha", type=float, default=1.0)
    parser.add_argument("--sid", type=int, default=1)
    parser.add_argument("--lid", type=int, default=1)
    parser.add_argument("--sent", type=str, default="sent.txt")
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--voc_config_file", type=str, required=True)
    parser.add_argument("--voc_checkpoint_file", type=str, required=True)
    args = parser.parse_args()

    if args.ngpu >= 1:
        device = "cuda"
    else:
        device = "cpu"

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build model
    model, train_args = TTSTask.build_model_from_file(args.train_config, args.model_file, device)
    model.to(dtype=getattr(torch, "float32")).eval()

    # 2. Tokenizer
    tokenizer = build_tokenizer(token_type="phn", g2p_type=args.g2p_type)
    token_id_converter = TokenIDConverter(token_list=args.token_list)

    # 3 .others config
    cfg = dict()
    cfg['use_teacher_forcing'] = False
    #cfg['use_att_constraint'] = False

    # 4 .Build vocoder
    with open(args.voc_config_file) as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    generator = Generator(h).to(device)
    
    state_dict_g = torch.load(args.voc_checkpoint_file, map_location=device)
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()

    with torch.no_grad():
        for idx, text in enumerate(Path(args.sent).open('r').readlines()):
            # prepare batch
            text = text.rstrip()
            batch = dict()
            tokens = tokenizer.text2tokens(text)
            text_ints = token_id_converter.tokens2ids(tokens)
            batch['text'] = torch.from_numpy(np.array(text_ints, dtype=np.int64))
            batch['sids'] = torch.LongTensor([args.sid])
            batch['lids'] = torch.LongTensor([args.lid])
            batch = {k: v.to(device) for k, v in batch.items()}

            # inference
            output_dict = model.inference(**batch, **cfg)
            feat_gen_denorm = output_dict['feat_gen_denorm']
            #feat_gen_denorm = feat_gen_denorm.detach().cpu().numpy()
            #np.save(output_dir / f"{idx}.npy", feat_gen_denorm)

            # apply vocoder (mel-to-wav)
            x = feat_gen_denorm.unsqueeze(0).transpose(1,2) # -> (1,80,T)
            y_g_hat = generator(x)
            audio = y_g_hat.squeeze()
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            
            write(output_dir / f"{idx}_pred.wav", h.sampling_rate, audio)


if __name__ == '__main__':
    main()
