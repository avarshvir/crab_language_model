# app.py
import streamlit as st
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import json
import time
import os

# Import our modular components
from frontend.ui_components import apply_custom_css, render_sidebar
from models.architecture import CRAB, LocalConfig 

# Page Config must be the first Streamlit command
st.set_page_config(page_title="CRAB AI", page_icon="🦀", layout="wide")
apply_custom_css()

@st.cache_resource
def load_crab_engine():
    """Loads the v2 QA weights securely into RAM."""
    try:
        # Load configuration
        with open("models/crab_config.json", "r") as f:
            cfg = LocalConfig(**json.load(f))
        
        # Build Model Chassis
        model = CRAB(cfg)
        
        # Inject weights (Use CPU map_location for local testing without GPU)
        state_dict = torch.load("models/crab_v2_qa.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict)
        model.eval() # Disable dropout
        
        # Load standard tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer, cfg, "🟢 CORE ONLINE"
    except Exception as e:
        return None, None, None, f"🔴 CORE OFFLINE: {str(e)}"

# Boot the engine
model, tokenizer, config, engine_status = load_crab_engine()

# Render Sidebar via Frontend UI
temperature, max_tokens = render_sidebar(engine_status, "70.3", "5.67")

# Main Interface
st.title("🦀 CRAB Intelligence")
st.markdown("Interact with the experimental `crab_v2_qa` model, built completely from scratch by Arshvir.")

user_prompt = st.text_area("Input Prompt:", placeholder="e.g., 'Who made you?' or 'What is your name?'", height=100)

if st.button("Initialize Generation"):
    if not model:
        st.error("Cannot generate: Model failed to load.")
    elif not user_input.strip():
        st.warning("Please provide an input sequence.")
    else:
        # Format explicitly for v2 QA
        formatted_prompt = f"[USER]: {user_prompt.strip()}\n[CRAB]: "
        idx = tokenizer.encode(formatted_prompt, return_tensors="pt")
        
        output_placeholder = st.empty()
        
        t0 = time.time()
        for _ in range(max_tokens):
            idx_cond = idx[:, -config.block_size:]
            with torch.no_grad():
                logits, _ = model(idx_cond)
            
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            
            if idx_next.item() == tokenizer.eos_token_id:
                break
                
            idx = torch.cat((idx, idx_next), dim=1)
            
            # Real-time streaming effect
            current_decode = tokenizer.decode(idx[0].tolist())
            response_only = current_decode.split("[CRAB]: ")[-1]
            output_placeholder.markdown(f"<div class='crab-response'>{response_only} ▌</div>", unsafe_allow_html=True)
            
        t1 = time.time()
        
        # Final clean output
        final_text = tokenizer.decode(idx[0].tolist()).split("[CRAB]: ")[-1]
        output_placeholder.markdown(f"<div class='crab-response'><b>Response:</b><br>{final_text}</div>", unsafe_allow_html=True)
        st.caption(f"⚡ Inference completed in {t1-t0:.2f} seconds.")