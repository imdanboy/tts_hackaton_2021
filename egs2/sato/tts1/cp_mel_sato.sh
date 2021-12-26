
mkdir -p mels
cat exp/tts_train_raw_phn_g2pk/inference_284epoch/eval1/denorm/feats.scp |
    while read -r line; do
        fpath=$(echo $line | cut -f2 -d' ')
        cp $fpath mels/
        #echo $fpath
    done
