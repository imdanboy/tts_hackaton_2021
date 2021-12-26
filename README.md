# TTS Hackaton 2021 
- http://ai문학음성데이터해커톤.kr/index.php

# demo link (google colab)
- https://colab.research.google.com/drive/15bYEn-206aLsp-huRVkuNpEXrjB4eqzz#scrollTo=sugmcGeOBFow

# model checkpoint link
- https://drive.google.com/drive/folders/1lhBO77GSROX_9dJN9L78PvUAj8n3yUGO?usp=sharing

# synthesized wavs (test sets)
- https://drive.google.com/drive/folders/1xBT31iCESMFZCmSZoGxR2pSzFy21t93Y?usp=sharing

## Simple Inference Demo
1. install required library
```
git clone https://github.com/imdanboy/tts_hackaton_2021.git
cd tts_hackaton_2021
pip install -e .
pip install g2pk
```

2. assert required model checkpoint file
```
# recommend absolute path (for easy use); change '/path/to' on your own
model_file = "/path/to/tts_hackaton_2021_ckpt/284epoch.pth"
voc_v1 = "/path/to/tts_hackaton_2021_ckpt/v1_g_00735000.ckpt"
voc_v2 = "/path/to/tts_hackaton_2021_ckpt/v2_g_00900000.ckpt"
```

3. build model
```
# change working directory first
!cd tts_hackaton_2021/egs2/sato/tts1

train_config = "./exp/tts_train_raw_phn_g2pk/config.yaml"
voc_config_file_v1 = "../../espnet2/hifi_gan/config_v1.json"
voc_config_file_v2 = "../../espnet2/hifi_gan/config_v2.json"
token_list = "./token_list/phn_g2pk/tokens.txt"

from espnet2.sato.demo import TextToSpeech

import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

tts_v1 = TextToSpeech(train_config=train_config, model_file=model_file, voc_config_file=voc_config_file_v1, voc_checkpoint_file=voc_v1, token_list=token_list, device=device)
tts_v2 = TextToSpeech(train_config=train_config, model_file=model_file, voc_config_file=voc_config_file_v2, voc_checkpoint_file=voc_v2, token_list=token_list, device=device)
```

4. tts (synthesize wavs from text)
```
# you can change working directory from now on.
text = "문학작품 낭송 음성데이터는 자연스럽게 문학작품을 낭송하는 인공지능 기술 개발을 위한 음성 데이터입니다."
sid = 14 # 27번 화자
lid = 3 # 무감정
tts_v1(text, sid=sid, lid=lid)
# "synthesis.wav" would be generated in current working directory
```

4.1 duration control
```
tts_v1(text, speed_control_alpha=0.7)
tts_v1(text, speed_control_alpha=1.3)
```
4.2 pitch control
```
tts_v1(text, f0_shift=50)
tts_v1(text, f0_shift=-50)
```

