# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Fastspeech2 related modules for ESPnet2."""

import logging

from typing import Dict
from typing import Optional
from typing import Sequence
from typing import Tuple

import torch
import torch.nn.functional as F

from typeguard import check_argument_types

from espnet.nets.pytorch_backend.conformer.encoder import (
    Encoder as ConformerEncoder,  # noqa: H301
)
from espnet.nets.pytorch_backend.fastspeech.duration_predictor import DurationPredictor
from espnet.nets.pytorch_backend.fastspeech.length_regulator import LengthRegulator
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.tacotron2.decoder import Postnet
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import (
    Encoder as TransformerEncoder,  # noqa: H301
)

from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.torch_utils.initialize import initialize
from espnet2.tts.abs_tts import AbsTTS
from espnet2.tts.fastspeech3.loss import FastSpeech3Loss
from espnet2.tts.fastspeech3.variance_predictor import VariancePredictor
from espnet2.tts.gst.style_encoder import StyleEncoder

import torch.nn as nn
from numba import jit
import numpy as np
from scipy.stats import betabinom

def gaussian_upsampling(dur, delta=0.1, mel_mask=None, inp_mask=None):
    # dur: (B,T_inp), mel_mask: (B,T_mel), inp_mask: (B,T_inp)
    B = dur.size(0)
    device = dur.device

    if mel_mask is None: # inference
        T_mel = dur.sum().round().int()
    else:
        T_mel = mel_mask.size(-1)
    t = torch.arange(0, T_mel).unsqueeze(0).repeat(B,1).to(device).float() # (B,T_mel)
    if mel_mask is not None:
        t = t * mel_mask.float()

    c = dur.cumsum(dim=-1) - dur/2 # (B,T_inp)

    # (B,T_mel,1) - (B,1,T_inp) = (B,T_mel,T_inp)
    energy = -1 * delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
    if inp_mask is not None:
        # (B,T_mel,T_inp)
        energy = energy.masked_fill(~(inp_mask.unsqueeze(1).repeat(1,T_mel,1)), -float("inf"))

    p_attn = torch.softmax(energy, dim=2) # (B,T_mel,T_inp)
    return p_attn # (B,T_mel,T_inp)

def forward_sum_loss(log_p_attn, inp_len, mel_len, blank_prob = np.e**-1):
    # log_p_attn: (B,T_mel,T_inp)
    B = log_p_attn.size(0)

    # add prior
    #bb_prior = betabinom_prior(inp_len, mel_len) # (B,T_mel,T_inp)
    bb_prior = BetaBinomPrior.get_p_attn_prior(inp_len, mel_len)
    bb_prior = bb_prior.to(dtype=log_p_attn.dtype, device=log_p_attn.device)
    log_p_attn = log_p_attn + bb_prior

    # a row must be added to the attention matrix to account for blank token of CTC loss
    # (B,T_mel,T_inp+1)
    log_p_attn_pd = F.pad(log_p_attn, (1,0,0,0,0,0), value=np.log(blank_prob))

    # construct target sequnece.
    # Every text token is mapped to a unique sequnece number.
    target_seq = torch.zeros_like(log_p_attn[:,0,:]) # (B,T_inp)
    for bidx in range(B):
        t_seq = torch.arange(1, inp_len[bidx]+1)
        target_seq[bidx, :inp_len[bidx]] = t_seq

    log_p_attn_pd = log_p_attn_pd.transpose(0,1) # -> (T_mel, B, T_inp+1)
    loss = F.ctc_loss(log_probs=log_p_attn_pd, targets=target_seq, input_lengths=mel_len,
            target_lengths=inp_len, zero_infinity=False)
    return loss

# log domain
# cache
class BetaBinomPrior:
    probs = {}

    @classmethod
    def get_p_attn_prior(cls, inp_len, mel_len, w=1):
        # w: scaling factor; lower -> wider the width
        B = len(inp_len)
        T_inp = inp_len.max()
        T_mel = mel_len.max()

        #bb_prior = torch.zeros(B,T_mel,T_inp)
        bb_prior = torch.full((B,T_mel,T_inp), fill_value=-np.inf)
        for bidx in range(B):
            T = mel_len[bidx].item()
            N = inp_len[bidx].item()
            key = str(T) + ',' + str(N)
            if key in cls.probs:
                prob = cls.probs[key]
            else:
                alpha = w * np.arange(1, T+1, dtype=float) # (T,)
                beta = w * np.array([T-t+1 for t in alpha])
                k = np.arange(N) # <- resonable? instead of (1,N+1)?
                batched_k = k[...,None] # (N,1)
                prob = betabinom.logpmf(batched_k, N, alpha, beta) # (N,T)
                cls.probs[key] = prob

            prob = torch.from_numpy(prob).transpose(0,1) # -> (T,N)
            bb_prior[bidx, :T, :N] = prob

        return bb_prior # (B,T_mel,T_inp)

@jit(nopython=True)
def average_var_by_dur(f, e, d, inp_len, mel_len, zero_d_value=0):
    # f,e: (B,T_mel), d: (B,T_inp)
    B = d.shape[0]
    T_inp = d.shape[0]

    f_avg = np.zeros_like(d) # (B,T_inp)
    e_avg = np.zeros_like(d) # (B,T_inp)
    #d = d.round().to(np.int)
    d = d.astype(np.int32)

    for b in range(B):
        t_inp = inp_len[b]
        t_mel = mel_len[b]
        cur_d = d[b, :t_inp]
        #d_cumsum = np.pad(cur_d.cumsum(axis=0), (1,0), 'constant', constant_values=0) # (t_inp+1,)
        d_cumsum = cur_d.cumsum()
        d_cumsum = [0] + list(d_cumsum) # (t_inp+1,)
        cur_f = f[b, :t_mel]
        cur_e = e[b, :t_mel]
        for n, (start, end) in enumerate(zip(d_cumsum[:-1], d_cumsum[1:])): # n: 0, ..., t_inp-1
            if len(cur_f[start:end]) != 0:
                f_avg[b,n] = cur_f[start:end].mean()
                e_avg[b,n] = cur_e[start:end].mean()
            else:
                f_avg[b,n] = zero_d_value
                e_avg[b,n] = zero_d_value
    return f_avg, e_avg

@jit(nopython=True)
def monotonic_alignment_search(log_p_attn):
    # log_p_attn: (T_mel, T_inp)

    T_mel = log_p_attn.shape[0]
    T_inp = log_p_attn.shape[1]
    Q = np.full((T_inp, T_mel), fill_value=-np.inf)

    log_prob = log_p_attn.transpose(1,0) # -> (T_inp,T_mel)
    # 1.  Q <- init first row for all j
    for j in range(T_mel):
        Q[0,j] = log_prob[0, :j+1].sum()

    # 2. 
    for j in range(1, T_mel):
        for i in range(1, min(j+1, T_inp)):
            Q[i,j] = max(Q[i-1,j-1], Q[i,j-1]) + log_prob[i,j]

    # 3.
    A = np.full((T_mel,), fill_value=T_inp-1)
    #for j in reversed(range(T_mel-1)): # T_mel-2, ..., 0
    for j in range(T_mel-2, -1, -1): # T_mel-2, ..., 0
        # 'i' in {A[j+1]-1, A[j+1]}
        i_a = A[j+1]-1
        i_b = A[j+1]
        if i_b == 0:
            argmax_i = 0
        elif Q[i_a,j] >= Q[i_b,j]:
            argmax_i = i_a
        else:
            argmax_i = i_b
        A[j] = argmax_i
    return A

def viterbi_decode(log_p_attn, ilens, olens):
    # log_p_attn: (B,T_feats,T_text)
    B = log_p_attn.size(0)
    T_text = log_p_attn.size(2)
    device = log_p_attn.device

    ds = torch.zeros((B,T_text), device=device)
    d_bin_loss = 0
    for b in range(B):
        cur_log_p_attn = log_p_attn[b, :olens[b], :ilens[b]]
        viterbi = monotonic_alignment_search(cur_log_p_attn.detach().cpu().numpy())

        d = np.bincount(viterbi)
        ds[b, :len(d)] = torch.from_numpy(d).to(device)

        t_idx = torch.arange(olens[b])
        # len(t_idx) == len(viterbi)
        d_bin_loss = d_bin_loss - cur_log_p_attn[t_idx, viterbi].mean() 
    d_bin_loss = d_bin_loss / B

    return ds, d_bin_loss

class TextMelAttention(nn.Module):
    def __init__(self, adim, odim):
        super(TextMelAttention, self).__init__()
        # text encoder
        self.t_conv1 = nn.Conv1d(adim, adim, kernel_size=3, padding=int((3-1)/2))
        self.t_conv2 = nn.Conv1d(adim, adim, kernel_size=1,padding=0)
        # mel encoder
        self.m_conv1 = nn.Conv1d(odim, adim, kernel_size=3, padding=1)
        self.m_conv2 = nn.Conv1d(adim, adim, kernel_size=3, padding=1)
        self.m_conv3 = nn.Conv1d(adim, adim, kernel_size=1, padding=0)

    def forward(self, hs, ys, mask=None):
        # hs:(B,T_text,adim), ys:(B,T_feats,odim)
        # mask: (B,1,T_inp) from inp_mask
        hs = hs.transpose(1,2) # -> (B,adim,T_text)
        hs = F.relu(self.t_conv1(hs))
        hs = self.t_conv2(hs)
        hs = hs.transpose(1,2) # -> (B,T_text,adim)

        ys = ys.transpose(1,2) # -> (B,odim,T_feats)
        ys = F.relu(self.m_conv1(ys))
        ys = F.relu(self.m_conv2(ys))
        ys = self.m_conv3(ys)
        ys = ys.transpose(1,2) # -> (B,T_feats,adim)

        # (B,T_feats,1,adim) - (B,1,T_text,adim) = (B,T_feats,T_text,adim)
        dist = ys.unsqueeze(2) - hs.unsqueeze(1)
        dist = torch.linalg.norm(dist, ord=2, dim=3) # (B,T_feats,T_text)
        score = -dist

        if mask is not None:
            score = score.masked_fill(mask, -np.inf)

        log_p_attn = F.log_softmax(score, dim=-1) # (B,T_feats,T_text)

        return log_p_attn # (B,T_feats,T_text)


class FastSpeech3(AbsTTS):
    """FastSpeech3 module.

    This is a module of FastSpeech2 described in `FastSpeech 2: Fast and
    High-Quality End-to-End Text to Speech`_. Instead of quantized pitch and
    energy, we use token-averaged value introduced in `FastPitch: Parallel
    Text-to-speech with Pitch Prediction`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558
    .. _`FastPitch: Parallel Text-to-speech with Pitch Prediction`:
        https://arxiv.org/abs/2006.06873

    """

    def __init__(
        self,
        # network structure related
        idim: int,
        odim: int,
        adim: int = 384,
        aheads: int = 4,
        elayers: int = 6,
        eunits: int = 1536,
        dlayers: int = 6,
        dunits: int = 1536,
        postnet_layers: int = 5,
        postnet_chans: int = 512,
        postnet_filts: int = 5,
        postnet_dropout_rate: float = 0.5,
        positionwise_layer_type: str = "conv1d",
        positionwise_conv_kernel_size: int = 1,
        use_scaled_pos_enc: bool = True,
        use_batch_norm: bool = True,
        encoder_normalize_before: bool = True,
        decoder_normalize_before: bool = True,
        encoder_concat_after: bool = False,
        decoder_concat_after: bool = False,
        reduction_factor: int = 1,
        encoder_type: str = "transformer",
        decoder_type: str = "transformer",
        transformer_enc_dropout_rate: float = 0.1,
        transformer_enc_positional_dropout_rate: float = 0.1,
        transformer_enc_attn_dropout_rate: float = 0.1,
        transformer_dec_dropout_rate: float = 0.1,
        transformer_dec_positional_dropout_rate: float = 0.1,
        transformer_dec_attn_dropout_rate: float = 0.1,
        # only for conformer
        conformer_rel_pos_type: str = "legacy",
        conformer_pos_enc_layer_type: str = "rel_pos",
        conformer_self_attn_layer_type: str = "rel_selfattn",
        conformer_activation_type: str = "swish",
        use_macaron_style_in_conformer: bool = True,
        use_cnn_in_conformer: bool = True,
        zero_triu: bool = False,
        conformer_enc_kernel_size: int = 7,
        conformer_dec_kernel_size: int = 31,
        # duration predictor
        duration_predictor_layers: int = 2,
        duration_predictor_chans: int = 384,
        duration_predictor_kernel_size: int = 3,
        duration_predictor_dropout_rate: float = 0.1,
        # energy predictor
        energy_predictor_layers: int = 2,
        energy_predictor_chans: int = 384,
        energy_predictor_kernel_size: int = 3,
        energy_predictor_dropout: float = 0.5,
        energy_embed_kernel_size: int = 9,
        energy_embed_dropout: float = 0.5,
        stop_gradient_from_energy_predictor: bool = False,
        # pitch predictor
        pitch_predictor_layers: int = 2,
        pitch_predictor_chans: int = 384,
        pitch_predictor_kernel_size: int = 3,
        pitch_predictor_dropout: float = 0.5,
        pitch_embed_kernel_size: int = 9,
        pitch_embed_dropout: float = 0.5,
        stop_gradient_from_pitch_predictor: bool = False,
        # extra embedding related
        spks: Optional[int] = None,
        langs: Optional[int] = None,
        spk_embed_dim: Optional[int] = None,
        spk_embed_integration_type: str = "add",
        use_gst: bool = False,
        gst_tokens: int = 10,
        gst_heads: int = 4,
        gst_conv_layers: int = 6,
        gst_conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        gst_conv_kernel_size: int = 3,
        gst_conv_stride: int = 2,
        gst_gru_layers: int = 1,
        gst_gru_units: int = 128,
        # training related
        init_type: str = "xavier_uniform",
        init_enc_alpha: float = 1.0,
        init_dec_alpha: float = 1.0,
        use_masking: bool = False,
        use_weighted_masking: bool = False,
    ):
        """Initialize FastSpeech2 module.

        Args:
            idim (int): Dimension of the inputs.
            odim (int): Dimension of the outputs.
            elayers (int): Number of encoder layers.
            eunits (int): Number of encoder hidden units.
            dlayers (int): Number of decoder layers.
            dunits (int): Number of decoder hidden units.
            postnet_layers (int): Number of postnet layers.
            postnet_chans (int): Number of postnet channels.
            postnet_filts (int): Kernel size of postnet.
            postnet_dropout_rate (float): Dropout rate in postnet.
            use_scaled_pos_enc (bool): Whether to use trainable scaled pos encoding.
            use_batch_norm (bool): Whether to use batch normalization in encoder prenet.
            encoder_normalize_before (bool): Whether to apply layernorm layer before
                encoder block.
            decoder_normalize_before (bool): Whether to apply layernorm layer before
                decoder block.
            encoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in encoder.
            decoder_concat_after (bool): Whether to concatenate attention layer's input
                and output in decoder.
            reduction_factor (int): Reduction factor.
            encoder_type (str): Encoder type ("transformer" or "conformer").
            decoder_type (str): Decoder type ("transformer" or "conformer").
            transformer_enc_dropout_rate (float): Dropout rate in encoder except
                attention and positional encoding.
            transformer_enc_positional_dropout_rate (float): Dropout rate after encoder
                positional encoding.
            transformer_enc_attn_dropout_rate (float): Dropout rate in encoder
                self-attention module.
            transformer_dec_dropout_rate (float): Dropout rate in decoder except
                attention & positional encoding.
            transformer_dec_positional_dropout_rate (float): Dropout rate after decoder
                positional encoding.
            transformer_dec_attn_dropout_rate (float): Dropout rate in decoder
                self-attention module.
            conformer_rel_pos_type (str): Relative pos encoding type in conformer.
            conformer_pos_enc_layer_type (str): Pos encoding layer type in conformer.
            conformer_self_attn_layer_type (str): Self-attention layer type in conformer
            conformer_activation_type (str): Activation function type in conformer.
            use_macaron_style_in_conformer: Whether to use macaron style FFN.
            use_cnn_in_conformer: Whether to use CNN in conformer.
            zero_triu: Whether to use zero triu in relative self-attention module.
            conformer_enc_kernel_size: Kernel size of encoder conformer.
            conformer_dec_kernel_size: Kernel size of decoder conformer.
            duration_predictor_layers (int): Number of duration predictor layers.
            duration_predictor_chans (int): Number of duration predictor channels.
            duration_predictor_kernel_size (int): Kernel size of duration predictor.
            duration_predictor_dropout_rate (float): Dropout rate in duration predictor.
            pitch_predictor_layers (int): Number of pitch predictor layers.
            pitch_predictor_chans (int): Number of pitch predictor channels.
            pitch_predictor_kernel_size (int): Kernel size of pitch predictor.
            pitch_predictor_dropout_rate (float): Dropout rate in pitch predictor.
            pitch_embed_kernel_size (float): Kernel size of pitch embedding.
            pitch_embed_dropout_rate (float): Dropout rate for pitch embedding.
            stop_gradient_from_pitch_predictor: Whether to stop gradient from pitch
                predictor to encoder.
            energy_predictor_layers (int): Number of energy predictor layers.
            energy_predictor_chans (int): Number of energy predictor channels.
            energy_predictor_kernel_size (int): Kernel size of energy predictor.
            energy_predictor_dropout_rate (float): Dropout rate in energy predictor.
            energy_embed_kernel_size (float): Kernel size of energy embedding.
            energy_embed_dropout_rate (float): Dropout rate for energy embedding.
            stop_gradient_from_energy_predictor: Whether to stop gradient from energy
                predictor to encoder.
            spks (Optional[int]): Number of speakers. If set to > 1, assume that the
                sids will be provided as the input and use sid embedding layer.
            langs (Optional[int]): Number of languages. If set to > 1, assume that the
                lids will be provided as the input and use sid embedding layer.
            spk_embed_dim (Optional[int]): Speaker embedding dimension. If set to > 0,
                assume that spembs will be provided as the input.
            spk_embed_integration_type: How to integrate speaker embedding.
            use_gst (str): Whether to use global style token.
            gst_tokens (int): The number of GST embeddings.
            gst_heads (int): The number of heads in GST multihead attention.
            gst_conv_layers (int): The number of conv layers in GST.
            gst_conv_chans_list: (Sequence[int]):
                List of the number of channels of conv layers in GST.
            gst_conv_kernel_size (int): Kernel size of conv layers in GST.
            gst_conv_stride (int): Stride size of conv layers in GST.
            gst_gru_layers (int): The number of GRU layers in GST.
            gst_gru_units (int): The number of GRU units in GST.
            init_type (str): How to initialize transformer parameters.
            init_enc_alpha (float): Initial value of alpha in scaled pos encoding of the
                encoder.
            init_dec_alpha (float): Initial value of alpha in scaled pos encoding of the
                decoder.
            use_masking (bool): Whether to apply masking for padded part in loss
                calculation.
            use_weighted_masking (bool): Whether to apply weighted masking in loss
                calculation.

        """
        assert check_argument_types()
        super().__init__()

        # store hyperparameters
        self.idim = idim
        self.odim = odim
        self.eos = idim - 1
        self.reduction_factor = reduction_factor
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.stop_gradient_from_pitch_predictor = stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = stop_gradient_from_energy_predictor
        self.use_scaled_pos_enc = use_scaled_pos_enc
        self.use_gst = use_gst

        # use idx 0 as padding idx
        self.padding_idx = 0

        # get positional encoding class
        pos_enc_class = (
            ScaledPositionalEncoding if self.use_scaled_pos_enc else PositionalEncoding
        )

        # check relative positional encoding compatibility
        if "conformer" in [encoder_type, decoder_type]:
            if conformer_rel_pos_type == "legacy":
                if conformer_pos_enc_layer_type == "rel_pos":
                    conformer_pos_enc_layer_type = "legacy_rel_pos"
                    logging.warning(
                        "Fallback to conformer_pos_enc_layer_type = 'legacy_rel_pos' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
                if conformer_self_attn_layer_type == "rel_selfattn":
                    conformer_self_attn_layer_type = "legacy_rel_selfattn"
                    logging.warning(
                        "Fallback to "
                        "conformer_self_attn_layer_type = 'legacy_rel_selfattn' "
                        "due to the compatibility. If you want to use the new one, "
                        "please use conformer_pos_enc_layer_type = 'latest'."
                    )
            elif conformer_rel_pos_type == "latest":
                assert conformer_pos_enc_layer_type != "legacy_rel_pos"
                assert conformer_self_attn_layer_type != "legacy_rel_selfattn"
            else:
                raise ValueError(f"Unknown rel_pos_type: {conformer_rel_pos_type}")

        # define encoder
        encoder_input_layer = torch.nn.Embedding(
            num_embeddings=idim, embedding_dim=adim, padding_idx=self.padding_idx
        )
        if encoder_type == "transformer":
            self.encoder = TransformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif encoder_type == "conformer":
            self.encoder = ConformerEncoder(
                idim=idim,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=eunits,
                num_blocks=elayers,
                input_layer=encoder_input_layer,
                dropout_rate=transformer_enc_dropout_rate,
                positional_dropout_rate=transformer_enc_positional_dropout_rate,
                attention_dropout_rate=transformer_enc_attn_dropout_rate,
                normalize_before=encoder_normalize_before,
                concat_after=encoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_enc_kernel_size,
                zero_triu=zero_triu,
            )
        else:
            raise ValueError(f"{encoder_type} is not supported.")

        # define GST
        if self.use_gst:
            self.gst = StyleEncoder(
                idim=odim,  # the input is mel-spectrogram
                gst_tokens=gst_tokens,
                gst_token_dim=adim,
                gst_heads=gst_heads,
                conv_layers=gst_conv_layers,
                conv_chans_list=gst_conv_chans_list,
                conv_kernel_size=gst_conv_kernel_size,
                conv_stride=gst_conv_stride,
                gru_layers=gst_gru_layers,
                gru_units=gst_gru_units,
            )

        # define spk and lang embedding
        self.spks = None
        if spks is not None and spks > 1:
            self.spks = spks
            self.sid_emb = torch.nn.Embedding(spks, adim)
        self.langs = None
        if langs is not None and langs > 1:
            self.langs = langs
            self.lid_emb = torch.nn.Embedding(langs, adim)

        # define additional projection for speaker embedding
        self.spk_embed_dim = None
        if spk_embed_dim is not None and spk_embed_dim > 0:
            self.spk_embed_dim = spk_embed_dim
            self.spk_embed_integration_type = spk_embed_integration_type
        if self.spk_embed_dim is not None:
            if self.spk_embed_integration_type == "add":
                self.projection = torch.nn.Linear(self.spk_embed_dim, adim)
            else:
                self.projection = torch.nn.Linear(adim + self.spk_embed_dim, adim)

        # define duration predictor
        self.duration_predictor = DurationPredictor(
            idim=adim,
            n_layers=duration_predictor_layers,
            n_chans=duration_predictor_chans,
            kernel_size=duration_predictor_kernel_size,
            dropout_rate=duration_predictor_dropout_rate,
        )

        # define pitch predictor
        self.pitch_predictor = VariancePredictor(
            idim=adim,
            n_layers=pitch_predictor_layers,
            n_chans=pitch_predictor_chans,
            kernel_size=pitch_predictor_kernel_size,
            dropout_rate=pitch_predictor_dropout,
        )
        # NOTE(kan-bayashi): We use continuous pitch + FastPitch style avg
        self.pitch_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=pitch_embed_kernel_size,
                padding=(pitch_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(pitch_embed_dropout),
        )

        # define energy predictor
        self.energy_predictor = VariancePredictor(
            idim=adim,
            n_layers=energy_predictor_layers,
            n_chans=energy_predictor_chans,
            kernel_size=energy_predictor_kernel_size,
            dropout_rate=energy_predictor_dropout,
        )
        # NOTE(kan-bayashi): We use continuous enegy + FastPitch style avg
        self.energy_embed = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=1,
                out_channels=adim,
                kernel_size=energy_embed_kernel_size,
                padding=(energy_embed_kernel_size - 1) // 2,
            ),
            torch.nn.Dropout(energy_embed_dropout),
        )

        # define length regulator
        #self.length_regulator = LengthRegulator()

        # define decoder
        # NOTE: we use encoder as decoder
        # because fastspeech's decoder is the same as encoder
        if decoder_type == "transformer":
            self.decoder = TransformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                pos_enc_class=pos_enc_class,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
            )
        elif decoder_type == "conformer":
            self.decoder = ConformerEncoder(
                idim=0,
                attention_dim=adim,
                attention_heads=aheads,
                linear_units=dunits,
                num_blocks=dlayers,
                input_layer=None,
                dropout_rate=transformer_dec_dropout_rate,
                positional_dropout_rate=transformer_dec_positional_dropout_rate,
                attention_dropout_rate=transformer_dec_attn_dropout_rate,
                normalize_before=decoder_normalize_before,
                concat_after=decoder_concat_after,
                positionwise_layer_type=positionwise_layer_type,
                positionwise_conv_kernel_size=positionwise_conv_kernel_size,
                macaron_style=use_macaron_style_in_conformer,
                pos_enc_layer_type=conformer_pos_enc_layer_type,
                selfattention_layer_type=conformer_self_attn_layer_type,
                activation_type=conformer_activation_type,
                use_cnn_module=use_cnn_in_conformer,
                cnn_module_kernel=conformer_dec_kernel_size,
            )
        else:
            raise ValueError(f"{decoder_type} is not supported.")

        # define final projection
        self.feat_out = torch.nn.Linear(adim, odim * reduction_factor)

        # define postnet
        self.postnet = (
            None
            if postnet_layers == 0
            else Postnet(
                idim=idim,
                odim=odim,
                n_layers=postnet_layers,
                n_chans=postnet_chans,
                n_filts=postnet_filts,
                use_batch_norm=use_batch_norm,
                dropout_rate=postnet_dropout_rate,
            )
        )

        # initialize parameters
        self._reset_parameters(
            init_type=init_type,
            init_enc_alpha=init_enc_alpha,
            init_dec_alpha=init_dec_alpha,
        )

        # define criterions
        self.criterion = FastSpeech3Loss(
            use_masking=use_masking, use_weighted_masking=use_weighted_masking
        )

        self.text_mel_attention = TextMelAttention(adim, odim)

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        pitch: torch.Tensor,
        pitch_lengths: torch.Tensor,
        energy: torch.Tensor,
        energy_lengths: torch.Tensor,
        durations: torch.Tensor = None,
        durations_lengths: torch.Tensor = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        joint_training: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Calculate forward propagation.

        Args:
            text (LongTensor): Batch of padded token ids (B, T_text).
            text_lengths (LongTensor): Batch of lengths of each input (B,).
            feats (Tensor): Batch of padded target features (B, T_feats, odim).
            feats_lengths (LongTensor): Batch of the lengths of each target (B,).
            durations (LongTensor): Batch of padded durations (B, T_text + 1).
            durations_lengths (LongTensor): Batch of duration lengths (B, T_text + 1).
            pitch (Tensor): Batch of padded token-averaged pitch (B, T_text + 1, 1).
            pitch_lengths (LongTensor): Batch of pitch lengths (B, T_text + 1).
            energy (Tensor): Batch of padded token-averaged energy (B, T_text + 1, 1).
            energy_lengths (LongTensor): Batch of energy lengths (B, T_text + 1).
            spembs (Optional[Tensor]): Batch of speaker embeddings (B, spk_embed_dim).
            sids (Optional[Tensor]): Batch of speaker IDs (B, 1).
            lids (Optional[Tensor]): Batch of language IDs (B, 1).
            joint_training (bool): Whether to perform joint training with vocoder.

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value if not joint training else model outputs.

        """
        text = text[:, : text_lengths.max()]  # for data-parallel
        feats = feats[:, : feats_lengths.max()]  # for data-parallel
        #durations = durations[:, : durations_lengths.max()]  # for data-parallel
        pitch = pitch[:, : pitch_lengths.max()]  # for data-parallel
        energy = energy[:, : energy_lengths.max()]  # for data-parallel

        batch_size = text.size(0)

        # Add eos at the last of sequence
        xs = F.pad(text, [0, 1], "constant", self.padding_idx)
        for i, l in enumerate(text_lengths):
            xs[i, l] = self.eos
        ilens = text_lengths + 1

        #ys, ds, ps, es = feats, durations, pitch, energy
        ys, ps, es = feats, pitch, energy
        olens = feats_lengths

        # forward propagation
        #before_outs, after_outs, d_outs, p_outs, e_outs = self._forward(
        before_outs, after_outs, d_outs, p_outs, e_outs, ps, es, ds, log_p_attn, d_bin_loss = self._forward(
            xs,
            ilens,
            ys,
            olens,
            ps=ps,
            es=es,
            spembs=spembs,
            sids=sids,
            lids=lids,
            is_inference=False,
        )
        ###
        align_loss = forward_sum_loss(log_p_attn, ilens, olens)
        ###

        # modify mod part of groundtruth
        if self.reduction_factor > 1:
            olens = olens.new([olen - olen % self.reduction_factor for olen in olens])
            max_olen = max(olens)
            ys = ys[:, :max_olen]

        # calculate loss
        if self.postnet is None:
            after_outs = None

        # calculate loss
        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
            after_outs=after_outs,
            before_outs=before_outs,
            d_outs=d_outs,
            p_outs=p_outs,
            e_outs=e_outs,
            ys=ys,
            ds=ds,
            ps=ps,
            es=es,
            ilens=ilens,
            olens=olens,
        )
        #loss = l1_loss + duration_loss + pitch_loss + energy_loss
        loss = l1_loss + duration_loss + pitch_loss + energy_loss + d_bin_loss + align_loss

        stats = dict(
            l1_loss=l1_loss.item(),
            duration_loss=duration_loss.item(),
            pitch_loss=pitch_loss.item(),
            energy_loss=energy_loss.item(),
            d_bin_loss=d_bin_loss.item(),
            align_loss=align_loss.item(),
        )

        # report extra information
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                encoder_alpha=self.encoder.embed[-1].alpha.data.item(),
            )
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            stats.update(
                decoder_alpha=self.decoder.embed[-1].alpha.data.item(),
            )

        if not joint_training:
            stats.update(loss=loss.item())
            loss, stats, weight = force_gatherable(
                (loss, stats, batch_size), loss.device
            )
            return loss, stats, weight
        else:
            return loss, stats, after_outs if after_outs is not None else before_outs

    def _forward(
        self,
        xs: torch.Tensor,
        ilens: torch.Tensor,
        ys: Optional[torch.Tensor] = None,
        olens: Optional[torch.Tensor] = None,
        ds: Optional[torch.Tensor] = None,
        ps: Optional[torch.Tensor] = None,
        es: Optional[torch.Tensor] = None,
        spembs: Optional[torch.Tensor] = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        is_inference: bool = False,
        alpha: float = 1.0,
        f0_shift = None,
        pitch_normalize = None,
    ) -> Sequence[torch.Tensor]:
        # forward encoder
        x_masks = self._source_mask(ilens)
        hs, _ = self.encoder(xs, x_masks)  # (B, T_text, adim)

        # integrate with GST
        if self.use_gst:
            style_embs = self.gst(ys)
            hs = hs + style_embs.unsqueeze(1)

        # integrate with SID and LID embeddings
        if self.spks is not None:
            sid_embs = self.sid_emb(sids.view(-1))
            hs = hs + sid_embs.unsqueeze(1)
        if self.langs is not None:
            lid_embs = self.lid_emb(lids.view(-1))
            hs = hs + lid_embs.unsqueeze(1)

        # integrate speaker embedding
        if self.spk_embed_dim is not None:
            hs = self._integrate_with_spk_embed(hs, spembs)

        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(ilens).to(xs.device)

        if self.stop_gradient_from_pitch_predictor:
            p_outs = self.pitch_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            p_outs = self.pitch_predictor(hs, d_masks.unsqueeze(-1))
        if self.stop_gradient_from_energy_predictor:
            e_outs = self.energy_predictor(hs.detach(), d_masks.unsqueeze(-1))
        else:
            e_outs = self.energy_predictor(hs, d_masks.unsqueeze(-1))

        if is_inference:
            d_outs = self.duration_predictor.inference(hs, d_masks)  # (B, T_text)
            # use prediction in inference
            ###
            if f0_shift is not None:
                p_outs, _ = pitch_normalize.inverse(p_outs)
                p_outs = torch.exp(p_outs) + f0_shift
                p_outs, _ = pitch_normalize(torch.log(p_outs))
            ###
            p_embs = self.pitch_embed(p_outs.transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(e_outs.transpose(1, 2)).transpose(1, 2)
            hs = hs + e_embs + p_embs
            #hs = self.length_regulator(hs, d_outs, alpha)  # (B, T_feats, adim)
            ###
            d_outs = d_outs * alpha
            ###
            gu_p_attn = gaussian_upsampling(d_outs.to(torch.float))
            hs = torch.matmul(gu_p_attn, hs)
            log_p_attn = None
            d_bin_loss = None
        else:
            ###
            log_p_attn = self.text_mel_attention(hs, ys, mask=d_masks.unsqueeze(-2))
            ds, d_bin_loss = viterbi_decode(log_p_attn, ilens, olens)
            ps = ps.detach().cpu().numpy()
            es = es.detach().cpu().numpy()
            ilens_ = ilens.detach().cpu().numpy()
            olens_ = olens.detach().cpu().numpy()
            ds_ = ds.detach().cpu().numpy()
            ps, es = average_var_by_dur(ps, es, ds_, ilens_, olens_)
            ps = torch.from_numpy(ps).to(xs.device).unsqueeze(-1)
            es = torch.from_numpy(es).to(xs.device).unsqueeze(-1)
            ###
            d_outs = self.duration_predictor(hs, d_masks)
            # use groundtruth in training
            p_embs = self.pitch_embed(ps.transpose(1, 2)).transpose(1, 2)
            e_embs = self.energy_embed(es.transpose(1, 2)).transpose(1, 2)
            hs = hs + e_embs + p_embs
            #hs = self.length_regulator(hs, ds.to(torch.long))  # (B, T_feats, adim)
            inp_mask = make_non_pad_mask(ilens).to(xs.device)
            mel_mask = make_non_pad_mask(olens).to(ys.device)
            gu_p_attn = gaussian_upsampling(ds, mel_mask=mel_mask, inp_mask=inp_mask)
            hs = torch.matmul(gu_p_attn, hs)

        # forward decoder
        if olens is not None and not is_inference:
            if self.reduction_factor > 1:
                olens_in = olens.new([olen // self.reduction_factor for olen in olens])
            else:
                olens_in = olens
            h_masks = self._source_mask(olens_in)
        else:
            h_masks = None
        zs, _ = self.decoder(hs, h_masks)  # (B, T_feats, adim)
        before_outs = self.feat_out(zs).view(
            zs.size(0), -1, self.odim
        )  # (B, T_feats, odim)

        # postnet -> (B, T_feats//r * r, odim)
        if self.postnet is None:
            after_outs = before_outs
        else:
            after_outs = before_outs + self.postnet(
                before_outs.transpose(1, 2)
            ).transpose(1, 2)

        return before_outs, after_outs, d_outs, p_outs, e_outs, ps, es, ds, log_p_attn, d_bin_loss

    def inference(
        self,
        text: torch.Tensor,
        feats: Optional[torch.Tensor] = None,
        durations: Optional[torch.Tensor] = None,
        spembs: torch.Tensor = None,
        sids: Optional[torch.Tensor] = None,
        lids: Optional[torch.Tensor] = None,
        pitch: Optional[torch.Tensor] = None,
        energy: Optional[torch.Tensor] = None,
        alpha: float = 1.0,
        use_teacher_forcing: bool = False,
        f0_shift = None,
        pitch_normalize = None,
    ) -> Dict[str, torch.Tensor]:
        """Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (T_text,).
            feats (Optional[Tensor): Feature sequence to extract style (N, idim).
            durations (Optional[Tensor): Groundtruth of duration (T_text + 1,).
            spembs (Optional[Tensor): Speaker embedding vector (spk_embed_dim,).
            sids (Optional[Tensor]): Speaker ID (1,).
            lids (Optional[Tensor]): Language ID (1,).
            pitch (Optional[Tensor]): Groundtruth of token-avg pitch (T_text + 1, 1).
            energy (Optional[Tensor]): Groundtruth of token-avg energy (T_text + 1, 1).
            alpha (float): Alpha to control the speed.
            use_teacher_forcing (bool): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.

        Returns:
            Dict[str, Tensor]: Output dict including the following items:
                * feat_gen (Tensor): Output sequence of features (T_feats, odim).
                * duration (Tensor): Duration sequence (T_text + 1,).
                * pitch (Tensor): Pitch sequence (T_text + 1,).
                * energy (Tensor): Energy sequence (T_text + 1,).

        """
        x, y = text, feats
        spemb, d, p, e = spembs, durations, pitch, energy

        # add eos at the last of sequence
        x = F.pad(x, [0, 1], "constant", self.eos)

        # setup batch axis
        ilens = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        if y is not None:
            ys = y.unsqueeze(0)
        if spemb is not None:
            spembs = spemb.unsqueeze(0)

        if use_teacher_forcing:
            # use groundtruth of duration, pitch, and energy
            ds, ps, es = d.unsqueeze(0), p.unsqueeze(0), e.unsqueeze(0)
            _, outs, d_outs, p_outs, e_outs = self._forward(
                xs,
                ilens,
                ys,
                ds=ds,
                ps=ps,
                es=es,
                spembs=spembs,
                sids=sids,
                lids=lids,
            )  # (1, T_feats, odim)
        else:
            #_, outs, d_outs, p_outs, e_outs = self._forward(
            before_outs, outs, d_outs, p_outs, e_outs, ps, es, ds, log_p_attn, d_bin_loss = self._forward(
                xs,
                ilens,
                ys,
                spembs=spembs,
                sids=sids,
                lids=lids,
                is_inference=True,
                alpha=alpha,
                f0_shift=f0_shift,
                pitch_normalize=pitch_normalize,
            )  # (1, T_feats, odim)

        return dict(
            feat_gen=outs[0],
            duration=d_outs[0],
            pitch=p_outs[0],
            energy=e_outs[0],
        )

    def _integrate_with_spk_embed(
        self, hs: torch.Tensor, spembs: torch.Tensor
    ) -> torch.Tensor:
        """Integrate speaker embedding with hidden states.

        Args:
            hs (Tensor): Batch of hidden state sequences (B, T_text, adim).
            spembs (Tensor): Batch of speaker embeddings (B, spk_embed_dim).

        Returns:
            Tensor: Batch of integrated hidden state sequences (B, T_text, adim).

        """
        if self.spk_embed_integration_type == "add":
            # apply projection and then add to hidden states
            spembs = self.projection(F.normalize(spembs))
            hs = hs + spembs.unsqueeze(1)
        elif self.spk_embed_integration_type == "concat":
            # concat hidden states with spk embeds and then apply projection
            spembs = F.normalize(spembs).unsqueeze(1).expand(-1, hs.size(1), -1)
            hs = self.projection(torch.cat([hs, spembs], dim=-1))
        else:
            raise NotImplementedError("support only add or concat.")

        return hs

    def _source_mask(self, ilens: torch.Tensor) -> torch.Tensor:
        """Make masks for self-attention.

        Args:
            ilens (LongTensor): Batch of lengths (B,).

        Returns:
            Tensor: Mask tensor for self-attention.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

        Examples:
            >>> ilens = [5, 3]
            >>> self._source_mask(ilens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 0, 0]]], dtype=torch.uint8)

        """
        x_masks = make_non_pad_mask(ilens).to(next(self.parameters()).device)
        return x_masks.unsqueeze(-2)

    def _reset_parameters(
        self, init_type: str, init_enc_alpha: float, init_dec_alpha: float
    ):
        # initialize parameters
        if init_type != "pytorch":
            initialize(self, init_type)

        # initialize alpha in scaled positional encoding
        if self.encoder_type == "transformer" and self.use_scaled_pos_enc:
            self.encoder.embed[-1].alpha.data = torch.tensor(init_enc_alpha)
        if self.decoder_type == "transformer" and self.use_scaled_pos_enc:
            self.decoder.embed[-1].alpha.data = torch.tensor(init_dec_alpha)
