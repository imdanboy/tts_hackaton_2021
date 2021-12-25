python -m espnet2.bin.tts_decode \
    --train_config exp/tts_train_raw_phn_g2pk/config.yaml \
    --model_file exp/tts_train_raw_phn_g2pk/152epoch.pth \
    --voc_config_file voc_exp/config_v2.json \
    --voc_checkpoint_file voc_exp/g_00780000 \
    --sid 26 --lid 3
