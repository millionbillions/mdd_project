"""
LingMDDModel — Mispronunciation Detection & Diagnosis for Vietnamese.

Architecture:
  SSL backbone [+ Pitch Encoder (optional)]
  → forward cross-attn  (Q=audio,       K,V=canonical)  → CTC head
  → reverse cross-attn  (Q=canonical,   K,V=audio)      → binary MDD head
  + Contrastive loss: correct phonemes → mdd_hidden close to canonical embedding

Loss = focal_ctc + bce_lambda * bce(binary_mdd) + contrastive_lambda * contrastive
Supports: wav2vec2-vn-250h, wav2vec2-100h, hubert-ls960.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
import transformers
from transformers.utils import ModelOutput

from mdd.criterion import FocalCTCLoss
from mdd.utils import PAD_ID
import mdd.utils as _mdd_utils  # for dynamic LEN_VOCAB lookup (set_anti_mode mutates it)

BLANK_ID = PAD_ID
IGNORE_VALUE = -100
BCE_POS_WEIGHT = 10.0          # legacy plain BCE knob; ignored when use_focal_bce=True
CONTRASTIVE_MARGIN = 0.3       # mispronounced phonemes pushed below this similarity
FOCAL_BCE_ALPHA = 0.75         # weight on positive (mispronounced) class
FOCAL_BCE_GAMMA = 2.0          # focusing parameter — down-weight easy negatives


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class MDDModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None          # CTC logits  (B, T_a, V)
    mdd_logits: Optional[torch.FloatTensor] = None      # binary MDD  (B, T_c)
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=256):
        super().__init__()
        self.register_buffer("pos_table", self._sinusoid_table(n_position, d_hid))

    def _sinusoid_table(self, n, d):
        def angle_vec(pos):
            return [pos / np.power(10000, 2 * (j // 2) / d) for j in range(d)]
        table = np.array([angle_vec(i) for i in range(n)])
        table[:, 0::2] = np.sin(table[:, 0::2])
        table[:, 1::2] = np.cos(table[:, 1::2])
        return torch.FloatTensor(table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, : x.size(1)].clone().detach()


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class PitchEncoder(nn.Module):
    """
    Encodes Kaldi pitch features (2D: pitch + NCCF, 100fps)
    into acoustic feature space at wav2vec2 rate (~50fps).

    Input:  [B, T_pitch, 2]
    Output: [B, T_pitch//2, d_model]
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 256, kernel_size=5, stride=2, padding=2),  # 100fps → 50fps
            nn.ReLU(),
        )
        self.proj = nn.Linear(256, d_model)

    def forward(self, pitch: torch.Tensor) -> torch.Tensor:
        x = pitch.transpose(1, 2)   # [B, 2, T_pitch]
        x = self.conv(x)             # [B, 256, T_pitch//2]
        x = x.transpose(1, 2)       # [B, T_pitch//2, 256]
        return self.proj(x)          # [B, T_pitch//2, d_model]


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

BACKBONE_CONFIGS = {
    "wav2vec2-vn":   "nguyenvulebinh/wav2vec2-base-vietnamese-250h",
    "wav2vec2-100h": "facebook/wav2vec2-base-100h",
    "hubert":        "facebook/hubert-base-ls960",
}


class LingMDDModel(transformers.Wav2Vec2PreTrainedModel):
    """
    SSL backbone + optional pitch encoder
    + forward cross-attn (CTC) + reverse cross-attn (binary MDD)
    + contrastive loss on canonical embedding vs mdd_hidden.
    """

    def __init__(self, config, backbone_type="wav2vec2-vn",
                 focal_alpha=0.99, focal_gamma=2, bce_lambda=1.0,
                 contrastive_lambda=0.1, use_pitch=False,
                 use_focal_bce=False,
                 focal_bce_alpha=FOCAL_BCE_ALPHA,
                 focal_bce_gamma=FOCAL_BCE_GAMMA):
        super().__init__(config)
        self.backbone_type = backbone_type
        self.bce_lambda = bce_lambda
        self.contrastive_lambda = contrastive_lambda
        self.use_pitch = use_pitch
        self.use_focal_bce = use_focal_bce
        self.focal_bce_alpha = focal_bce_alpha
        self.focal_bce_gamma = focal_bce_gamma

        if backbone_type == "hubert":
            self.backbone = transformers.HubertModel(config)
        else:
            self.backbone = transformers.Wav2Vec2Model(config)

        hidden = config.hidden_size  # 768

        if use_pitch:
            self.pitch_encoder = PitchEncoder(hidden)

        self.ling_emb = nn.Embedding(config.vocab_size, hidden, padding_idx=BLANK_ID)
        self.pos_enc = PositionalEncoding(hidden, n_position=256)

        # Forward cross-attn: Q=audio, K,V=canonical → CTC
        self.attn_norm = RMSNorm(hidden)
        self.mha = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=hidden // 64, dropout=0.1, batch_first=True,
        )

        self.ffn_layer = nn.Sequential(
            RMSNorm(hidden),
            nn.Linear(hidden, hidden * 2),
            SwiGLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
        )
        self.out_norm = RMSNorm(hidden)
        self.lm_head = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden, config.vocab_size))

        # Reverse cross-attn: Q=canonical, K,V=audio → binary MDD per canonical phoneme
        self.mdd_attn_rev = nn.MultiheadAttention(
            embed_dim=hidden, num_heads=hidden // 64, dropout=0.1, batch_first=True,
        )
        self.mdd_norm = RMSNorm(hidden)
        self.mdd_head = nn.Linear(hidden, 1)

        self.focal_ctc = FocalCTCLoss(alpha=focal_alpha, gamma=focal_gamma, blank=BLANK_ID)

        self.post_init()

    @classmethod
    def from_backbone(cls, backbone_type, focal_alpha=0.99, focal_gamma=2,
                      spec_augment=True, bce_lambda=1.0,
                      contrastive_lambda=0.1, use_pitch=False,
                      use_focal_bce=False,
                      focal_bce_alpha=FOCAL_BCE_ALPHA,
                      focal_bce_gamma=FOCAL_BCE_GAMMA):
        backbone_id = BACKBONE_CONFIGS[backbone_type]

        # Read LEN_VOCAB at call time so set_anti_mode() (called before from_backbone)
        # takes effect. Importing as `from mdd.utils import LEN_VOCAB` would snapshot
        # the value at module import and miss any subsequent toggle.
        config_kwargs = {"vocab_size": _mdd_utils.LEN_VOCAB, "ignore_mismatched_sizes": True}
        if spec_augment:
            config_kwargs.update({
                "mask_time_prob": 0.025, "mask_time_length": 10,
                "mask_feature_prob": 0.001, "mask_feature_length": 16,
            })

        if backbone_type == "hubert":
            config = transformers.HubertConfig.from_pretrained(backbone_id, **config_kwargs)
        else:
            config = transformers.Wav2Vec2Config.from_pretrained(backbone_id, **config_kwargs)

        model = cls(config, backbone_type=backbone_type,
                    focal_alpha=focal_alpha, focal_gamma=focal_gamma,
                    bce_lambda=bce_lambda,
                    contrastive_lambda=contrastive_lambda,
                    use_pitch=use_pitch,
                    use_focal_bce=use_focal_bce,
                    focal_bce_alpha=focal_bce_alpha,
                    focal_bce_gamma=focal_bce_gamma)

        if backbone_type == "hubert":
            pretrained = transformers.HubertModel.from_pretrained(backbone_id)
        else:
            pretrained = transformers.Wav2Vec2Model.from_pretrained(backbone_id)
        model.backbone.load_state_dict(pretrained.state_dict(), strict=False)

        return model

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        canonical_labels: Optional[torch.Tensor] = None,
        binary_labels: Optional[torch.Tensor] = None,
        pitch_features: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        # --- Acoustic encoder ---
        backbone_out = self.backbone(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        audio_states = backbone_out[0]  # (B, T_a, 768)

        # --- Pitch fusion (optional) ---
        if self.use_pitch and pitch_features is not None:
            pitch_enc = self.pitch_encoder(pitch_features)  # [B, T_p, hidden]
            T_audio = audio_states.size(1)
            T_pitch = pitch_enc.size(1)
            if T_pitch != T_audio:
                # Linear interpolation to match wav2vec2 temporal resolution
                pitch_enc = F.interpolate(
                    pitch_enc.transpose(1, 2),
                    size=T_audio,
                    mode='linear',
                    align_corners=False,
                ).transpose(1, 2)
            audio_states = audio_states + pitch_enc

        # --- Linguistic encoder ---
        ling_states = self.pos_enc(self.ling_emb(canonical_labels))  # (B, T_c, 768)
        ling_states = self.attn_norm(ling_states)

        # --- Forward cross-attention: audio queries canonical → CTC ---
        hidden, _ = self.mha(audio_states, ling_states, ling_states, need_weights=False)
        hidden = hidden + audio_states
        hidden = self.ffn_layer(hidden) + hidden
        hidden = self.out_norm(hidden)
        logits = self.lm_head(hidden)  # (B, T_a, V)

        # --- Reverse cross-attention: canonical queries audio → binary MDD ---
        mdd_hidden, _ = self.mdd_attn_rev(ling_states, hidden, hidden, need_weights=False)
        mdd_hidden = self.mdd_norm(mdd_hidden + ling_states)  # residual
        mdd_logits = self.mdd_head(mdd_hidden).squeeze(-1)    # (B, T_c)

        # --- Losses ---
        loss = None
        if labels is not None:
            if attention_mask is None:
                attention_mask = torch.ones_like(input_values, dtype=torch.long)
            input_lengths = self._get_feat_extract_output_lengths(
                attention_mask.sum(-1)
            ).to(torch.long)

            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            log_probs = F.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
            with torch.backends.cudnn.flags(enabled=False):
                loss = self.focal_ctc(log_probs, flattened_targets, input_lengths, target_lengths)

            # Binary MDD BCE loss (plain BCE w/ pos_weight, or focal BCE)
            if binary_labels is not None and self.bce_lambda > 0:
                mask = binary_labels >= 0
                if mask.any():
                    logits_m = mdd_logits[mask]
                    labels_m = binary_labels[mask]
                    if self.use_focal_bce:
                        # Focal BCE: ɑ_t · (1-p_t)^γ · BCE
                        bce_per = F.binary_cross_entropy_with_logits(
                            logits_m, labels_m, reduction="none"
                        )
                        p_t = torch.exp(-bce_per)
                        focal_factor = (1.0 - p_t) ** self.focal_bce_gamma
                        alpha_t = torch.where(
                            labels_m > 0.5,
                            torch.full_like(labels_m, self.focal_bce_alpha),
                            torch.full_like(labels_m, 1.0 - self.focal_bce_alpha),
                        )
                        bce_loss = (alpha_t * focal_factor * bce_per).mean()
                    else:
                        pos_weight = torch.tensor(BCE_POS_WEIGHT, device=mdd_logits.device)
                        bce_loss = F.binary_cross_entropy_with_logits(
                            logits_m, labels_m, pos_weight=pos_weight,
                        )
                    loss = loss + self.bce_lambda * bce_loss

            # Contrastive loss: correct → mdd_hidden near canonical; mispronounced → far
            if binary_labels is not None and self.contrastive_lambda > 0:
                mask = binary_labels >= 0
                if mask.any():
                    h = F.normalize(mdd_hidden[mask], dim=-1)
                    c = F.normalize(ling_states[mask], dim=-1)
                    cos_sim = (h * c).sum(dim=-1)
                    labels_flat = binary_labels[mask]
                    cont_loss = (
                        labels_flat * F.relu(cos_sim - CONTRASTIVE_MARGIN)
                        + (1.0 - labels_flat) * (1.0 - cos_sim)
                    )
                    loss = loss + self.contrastive_lambda * cont_loss.mean()

        if not return_dict:
            output = (logits, mdd_logits) + backbone_out[2:]
            return ((loss,) + output) if loss is not None else output

        return MDDModelOutput(
            loss=loss,
            logits=logits,
            mdd_logits=mdd_logits,
            hidden_states=backbone_out.hidden_states if output_hidden_states else None,
            attentions=backbone_out.attentions if output_attentions else None,
        )
