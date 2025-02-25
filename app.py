import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import sentencepiece as spm
import math
from typing import List, Tuple
from dataclasses import dataclass

# Model Configuration
@dataclass
class ModelConfig:
    encoder_vocab_size: int
    decoder_vocab_size: int
    d_embed: int
    d_ff: int
    h: int
    N_encoder: int
    N_decoder: int
    max_seq_len: int
    dropout: float

# Model Architecture Classes
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_embed, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_embed % h == 0
        self.d_k = d_embed//h
        self.d_embed = d_embed
        self.h = h
        self.WQ = nn.Linear(d_embed, d_embed)
        self.WK = nn.Linear(d_embed, d_embed)
        self.WV = nn.Linear(d_embed, d_embed)
        self.linear = nn.Linear(d_embed, d_embed)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, x_query, x_key, x_value, mask=None):
        nbatch = x_query.size(0)
        query = self.WQ(x_query).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        key = self.WK(x_key).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        value = self.WV(x_value).view(nbatch, -1, self.h, self.d_k).transpose(1,2)
        
        scores = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        self.attention_weights = F.softmax(scores, dim=-1)
        p_atten = self.dropout(self.attention_weights)
        
        x = torch.matmul(p_atten, value)
        x = x.transpose(1, 2).contiguous().view(nbatch, -1, self.d_embed)
        return self.linear(x)

class ResidualConnection(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, sublayer):
        return x + self.drop(sublayer(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, config):
        super(EncoderBlock, self).__init__()
        self.atten = MultiHeadedAttention(config.h, config.d_embed, config.dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residual1 = ResidualConnection(config.d_embed, config.dropout)
        self.residual2 = ResidualConnection(config.d_embed, config.dropout)

    def forward(self, x, mask=None):
        x = self.residual1(x, lambda x: self.atten(x, x, x, mask=mask))
        return self.residual2(x, self.feed_forward)

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.encoder_vocab_size, config.d_embed) 
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed)) 
        self.encoder_blocks = nn.ModuleList([EncoderBlock(config) for _ in range(config.N_encoder)])
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.d_embed)

    def forward(self, input, mask=None):
        x = self.tok_embed(input)
        x_pos = self.pos_embed[:, :x.size(1), :]
        x = self.dropout(x + x_pos)
        for layer in self.encoder_blocks:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.atten1 = MultiHeadedAttention(config.h, config.d_embed)
        self.atten2 = MultiHeadedAttention(config.h, config.d_embed)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_embed, config.d_ff),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_embed)
        )
        self.residuals = nn.ModuleList([ResidualConnection(config.d_embed, config.dropout) 
                                       for i in range(3)])

    def forward(self, memory, src_mask, decoder_layer_input, trg_mask):
        x = memory
        y = decoder_layer_input
        y = self.residuals[0](y, lambda y: self.atten1(y, y, y, mask=trg_mask))
        y = self.residuals[1](y, lambda y: self.atten2(y, x, x, mask=src_mask))
        return self.residuals[2](y, self.feed_forward)

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_embed = config.d_embed
        self.tok_embed = nn.Embedding(config.decoder_vocab_size, config.d_embed)
        self.pos_embed = nn.Parameter(torch.zeros(1, config.max_seq_len, config.d_embed)) 
        self.dropout = nn.Dropout(config.dropout)
        self.decoder_blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.N_decoder)])
        self.norm = nn.LayerNorm(config.d_embed)
        self.linear = nn.Linear(config.d_embed, config.decoder_vocab_size)
    
    def future_mask(self, seq_len):
        mask = (torch.triu(torch.ones(seq_len, seq_len), diagonal=1) != 0)
        return mask.unsqueeze(0).unsqueeze(1)

    def forward(self, memory, src_mask, trg, trg_pad_mask):
        seq_len = trg.size(1)
        trg_mask = torch.logical_or(trg_pad_mask, self.future_mask(seq_len).to(trg.device))
        x = self.tok_embed(trg) + self.pos_embed[:, :trg.size(1), :]
        x = self.dropout(x)
        for layer in self.decoder_blocks:
            x = layer(memory, src_mask, x, trg_mask)
        x = self.norm(x)
        logits = self.linear(x)
        return logits

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, src_mask, trg, trg_pad_mask):
        return self.decoder(self.encoder(src, src_mask), src_mask, trg, trg_pad_mask)

# Constants
UNK, BOS, EOS, PAD = 0, 1, 2, 3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Streamlit UI and functionality
st.set_page_config(page_title="English-Urdu Translator", layout="wide")

st.markdown("""
<style>
    .chat-container {
        margin-bottom: 20px;
        
    }
    .en-message {
        background-color: #161819;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 80%;
        float: left;
        clear: both;
    }
    .ur-message {
        background-color: #161819;
        padding: 10px 15px;
        border-radius: 15px;
        margin: 5px 0;
        max-width: 80%;
        float: right;
        clear: both;
        text-align: right;
        direction: rtl;
    }
    .stTextInput>div>div>input {
        padding: 15px;
    }
    .stButton>button {
        width: 100%;
    }
    .submit-button {
        margin-top: 10px;
    }
</style>
""", unsafe_allow_html=True)

if 'messages' not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def load_models():
    """Load the translation model and tokenizers"""
    try:
        # Load SentencePiece models
        en_sp = spm.SentencePieceProcessor()
        ur_sp = spm.SentencePieceProcessor()
        en_sp.load('UMC005_en.model')
        ur_sp.load('UMC005_ur.model')
        
        # Load the entire model directly
        model = torch.load('final_model.pt', map_location=DEVICE)
        model.eval()
        
        return model, en_sp, ur_sp
    
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        raise e

def translate_text(text: str, model: torch.nn.Module, en_sp: spm.SentencePieceProcessor, 
                  ur_sp: spm.SentencePieceProcessor, max_len: int = 100) -> str:
    """Translate English text to Urdu"""
    model.eval()
    
    # Tokenize input
    src_ids = torch.tensor([[BOS] + en_sp.encode_as_ids(text) + [EOS]]).to(DEVICE)
    src_pad_mask = (src_ids == PAD).unsqueeze(1).unsqueeze(2)
    
    with torch.no_grad():
        encoder_output = model.encoder(src_ids, src_pad_mask)
        trg_ids = torch.tensor([[BOS]]).to(DEVICE)
        
        for _ in range(max_len):
            trg_pad_mask = (trg_ids == PAD).unsqueeze(1).unsqueeze(2)
            output = model.decoder(encoder_output, src_pad_mask, trg_ids, trg_pad_mask)
            next_token = output[:, -1:].argmax(dim=-1)
            trg_ids = torch.cat([trg_ids, next_token], dim=1)
            
            if next_token.item() == EOS:
                break
    
    # Decode the translation
    translation = ur_sp.decode_ids(trg_ids[0].cpu().tolist()[1:-1])
    return translation

def main():
    st.title("English-Urdu Translator")
    
    try:
        # Load models
        model, en_sp, ur_sp = load_models()
        
         # Create a form for input
        with st.form(key="translation_form"):
            input_text = st.text_area("Enter English text:", height=100)
            submit_button = st.form_submit_button("Translate", use_container_width=True)
            
            if submit_button and input_text.strip():
                # Add input to message history
                st.session_state.messages.append(("en", input_text))
                
                # Translate
                translation = translate_text(input_text, model, en_sp, ur_sp)
                
                if translation:
                    # Add translation to message history
                    st.session_state.messages.append(("ur", translation))

        # Display message history
        for lang, message in st.session_state.messages:
            if lang == "en":
                st.markdown(f'<div class="chat-container"><div class="en-message">{message}</div></div>', 
                        unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-container"><div class="ur-message">{message}</div></div>', 
                        unsafe_allow_html=True)

        # Add a clear button below the chat
        if st.session_state.messages:
            if st.button("Clear Chat"):
                st.session_state.messages = []
                st.experimental_rerun()
                
    except Exception as e:
        st.error(f"Error in translation process: {str(e)}")

if __name__ == "__main__":
    main()