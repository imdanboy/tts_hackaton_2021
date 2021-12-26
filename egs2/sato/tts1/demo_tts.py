train_config = "exp/tts_train_raw_phn_g2pk/config.yaml"
model_file = "exp/tts_train_raw_phn_g2pk/284epoch.pth"
voc_config_file = "voc_exp/hifigan_v2/config_v2.json"
voc_checkpoint_file = "voc_exp/hifigan_v2/g_00900000"
token_list = "token_list/phn_g2pk/tokens.txt"

from espnet2.sato.demo import TextToSpeech
tts = TextToSpeech(train_config, model_file, voc_config_file, voc_checkpoint_file, token_list)
alpha = 1.0
f0_shift = None
tts("문학작품 낭송 음성데이터는 자연스럽게 문학작품을 낭송하는 인공지능 기술 개발을 위한 음성 데이터입니다.", speed_control_alpha=alpha, f0_shift=f0_shift)
