from pathlib import Path
import argparse
import yaml
import json

import numpy as np
import torch
from scipy.io.wavfile import write

from espnet2.tasks.tts import TTSTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter

from espnet2.hifi_gan.models import Generator
from espnet2.hifi_gan.meldataset import MAX_WAV_VALUE
from espnet2.hifi_gan.env import AttrDict



class TextToSpeech:

    def __init__(self,
            train_config,
            model_file,
            voc_config_file,
            voc_checkpoint_file,
            token_list,
            g2p_type="g2pk",
            device="cpu",
        ):
        self.device = device
        # 1. Build mel-generator
        model, train_args = TTSTask.build_model_from_file(train_config, model_file, device)
        model.to(dtype=getattr(torch, "float32")).eval()
        self.model = model

        # 2. Build Tokenizer
        self.tokenizer = build_tokenizer(token_type="phn", g2p_type=g2p_type)
        self.token_id_converter = TokenIDConverter(token_list=token_list)

        # 3. other config for model inference
        cfg = {}
        cfg['use_teacher_forcing'] = False
        self.cfg = cfg

        # 4. Build vocoder
        with open(voc_config_file, 'r') as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        generator = Generator(h).to(device)

        state_dict_g = torch.load(voc_checkpoint_file, map_location=device)
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()
        self.generator = generator

        # others...
        self.sr = h.sampling_rate

    @torch.no_grad()
    def __call__(self,
        text,
        sid=14,
        lid=3,
        speed_control_alpha=1.0,
        ):
        # prepare batch
        tokens = self.tokenizer.text2tokens(text)
        text_ints = self.token_id_converter.tokens2ids(tokens)
        batch = dict()
        batch['text'] = torch.from_numpy(np.array(text_ints, dtype=np.int64))
        batch['sids'] = torch.LongTensor([sid])
        batch['lids'] = torch.LongTensor([lid])
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # inference
        output_dict = self.model.inference(**batch, **self.cfg)
        feat_gen_denorm = output_dict['feat_gen_denorm']

        x = feat_gen_denorm.unsqueeze(0).transpose(1,2) # -> (1,80,T)
        y_g_hat = self.generator(x)
        audio = y_g_hat.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().numpy().astype('int16')

        write(f"synthesis.wav", self.sr, audio)
        print("synthesis.wav generated.")
