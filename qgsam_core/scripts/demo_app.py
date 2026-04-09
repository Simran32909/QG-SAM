"""
QG-SAM Interactive Demo
Run with: streamlit run qgsam_core/scripts/demo_app.py
"""
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
os.chdir(repo_root)

import torch
import numpy as np
import torch.nn.functional as F
import streamlit as st
from PIL import Image
import torchvision.transforms as T
import matplotlib.cm as cm

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QG-SAM · Visual Question Answering",
    layout="wide",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
    padding: 1.8rem 2.2rem;
    border-radius: 12px;
    margin-bottom: 1.8rem;
}
.header h1 { color: #f1f5f9; font-size: 1.9rem; font-weight: 700; margin: 0; }
.header p  { color: #94a3b8; font-size: 0.95rem; margin: 0.4rem 0 0; }

.answer-card {
    background: linear-gradient(135deg, #0f3460, #1e40af);
    border-radius: 10px;
    padding: 1.4rem 1.8rem;
    text-align: center;
    margin-bottom: 1.2rem;
}
.answer-card .label { color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .08em; }
.answer-card .value { color: #7dd3fc; font-size: 2.2rem; font-weight: 700; margin-top: .3rem; }
.answer-card .conf  { color: #64748b; font-size: 0.85rem; margin-top: .2rem; }

.step-item { color: #cbd5e1; font-size: 0.92rem; margin: .35rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header">
  <h1>QG-SAM &mdash; Question-Grounded Visual Question Answering</h1>
  <p>The model identifies which image region serves as "evidence" before predicting an answer.</p>
</div>
""", unsafe_allow_html=True)

# ── Model loader ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model weights...")
def load_model(ckpt_path, questions_json, config_path):
    import yaml
    from qgsam_core.models import QGSAM
    from qgsam_core.train  import QGSAMLightning
    from qgsam_core.data   import GQADataset

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    d_cfg, m_cfg = cfg["data"], cfg["model"]

    ds = GQADataset(
        questions_path=questions_json,
        images_dir=d_cfg["gqa_images"],
        top_k_answers=int(d_cfg.get("top_k_answers", 1000)),
        unknown_answer_token=str(d_cfg.get("unknown_answer_token", "unknown")),
        max_samples=None,
        is_train=False,
    )

    model = QGSAM(
        num_answers=len(ds.idx_to_answer),
        hidden_size=m_cfg.get("hidden_size", 512),
        image_feat_dim=m_cfg.get("image_feat_dim", 768),
        question_dim=m_cfg.get("question_dim", 768),
        num_masks=m_cfg.get("num_masks", 4),
        num_heads=m_cfg.get("num_heads", 8),
        num_cross_layers=m_cfg.get("num_cross_layers", 1),
        num_reasoner_layers=m_cfg.get("num_reasoner_layers", 2),
        dropout=m_cfg.get("dropout", 0.1),
        clip_model=m_cfg.get("clip_model", "openai/clip-vit-base-patch16"),
    )
    pl_model = QGSAMLightning.load_from_checkpoint(
        ckpt_path, model=model,
        num_answers=len(ds.idx_to_answer),
        idx_to_answer=ds.idx_to_answer,
        strict=False,
    )
    pl_model.eval()
    return pl_model.to("cpu"), ds.idx_to_answer, "cpu"


def preprocess(pil_img, size=224):
    tf = T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(pil_img.convert("RGB")).unsqueeze(0)


def make_heatmap(pil_img, heat, alpha=0.5):
    img  = np.array(pil_img.convert("RGB").resize((224, 224))).astype(float) / 255.0
    norm = (heat - heat.min()) / (heat.max() - heat.min() + 1e-8)
    rgb  = cm.inferno(norm)[:, :, :3]
    out  = np.clip((1 - alpha) * img + alpha * rgb, 0, 1)
    return Image.fromarray((out * 255).astype(np.uint8))


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Configuration")
    ckpt_path = st.text_input(
        "Checkpoint",
        value="qgsam_core/logs/checkpoints/epochepoch=02_vlossval_loss=2.0999_20260406_235729.ckpt",
    )
    questions_json = st.text_input(
        "Questions JSON (vocabulary source)",
        value="gqa_data/train_balanced_questions_with_boxes.json",
    )
    config_path = st.text_input(
        "Config YAML",
        value="qgsam_core/configs/default.yaml",
    )
    alpha = st.slider("Heatmap Opacity", 0.1, 0.9, 0.45, 0.05)
    st.caption("Inference device: CPU")

# ── Main layout ───────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("#### Input Image")
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        pil_img = Image.open(uploaded).convert("RGB")
        st.image(pil_img, use_container_width=True)

with right:
    st.markdown("#### Question")
    question = st.text_input("", placeholder="e.g. What color is the car?")
    run = st.button("Run Inference", type="primary", use_container_width=True)

# ── Inference ─────────────────────────────────────────────────────────────────
if run:
    if not uploaded:
        st.warning("Upload an image first.")
    elif not question.strip():
        st.warning("Enter a question.")
    else:
        status = st.status("Running inference pipeline...", expanded=True)
        try:
            with status:
                st.markdown('<p class="step-item">Step 1 / 5 &mdash; Loading model checkpoint</p>', unsafe_allow_html=True)
                prog = st.progress(0)
                model, idx_to_answer, device = load_model(ckpt_path, questions_json, config_path)
                prog.progress(20)

                st.markdown('<p class="step-item">Step 2 / 5 &mdash; Tokenizing question with BERT</p>', unsafe_allow_html=True)
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained("bert-base-uncased")(
                    question.strip(), return_tensors="pt",
                    padding=True, truncation=True, max_length=64,
                )
                input_ids      = tok["input_ids"].to(device)
                attention_mask = tok["attention_mask"].to(device)
                prog.progress(40)

                st.markdown('<p class="step-item">Step 3 / 5 &mdash; Encoding image with CLIP ViT</p>', unsafe_allow_html=True)
                img_t = preprocess(pil_img).to(device)
                prog.progress(55)

                st.markdown('<p class="step-item">Step 4 / 5 &mdash; Cross-attention: generating evidence masks</p>', unsafe_allow_html=True)
                with torch.no_grad():
                    out = model.model(img_t, input_ids, attention_mask)
                prog.progress(80)

                st.markdown('<p class="step-item">Step 5 / 5 &mdash; Reasoning over evidence, predicting answer</p>', unsafe_allow_html=True)
                logits  = out["answer_logits"][0]
                probs   = torch.softmax(logits, dim=-1)
                topk    = torch.topk(probs, k=min(5, len(idx_to_answer)))
                answers = [(idx_to_answer[i], probs[i].item()) for i in topk.indices]

                mask_probs = torch.sigmoid(out["mask_logits"])
                heat, _    = mask_probs[0].max(dim=0)
                heat = F.interpolate(
                    heat.unsqueeze(0).unsqueeze(0),
                    size=(224, 224), mode="bilinear", align_corners=False
                )[0, 0].cpu().numpy()
                prog.progress(100)

            status.update(label="Inference complete.", state="complete", expanded=False)

            # Answer card
            top_ans, top_conf = answers[0]
            st.markdown(f"""
            <div class="answer-card">
              <div class="label">Predicted Answer</div>
              <div class="value">{top_ans.upper()}</div>
              <div class="conf">Confidence &nbsp; {top_conf*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # Heatmap + top-5
            col_a, col_b = st.columns([1, 1], gap="large")
            with col_a:
                st.markdown("#### Evidence Region")
                st.caption("Pixels the model attended to when forming its answer.")
                st.image(make_heatmap(pil_img, heat, alpha), use_container_width=True)

            with col_b:
                st.markdown("#### Top 5 Predictions")
                for rank, (ans, conf) in enumerate(answers):
                    pct   = int(conf * 100)
                    color = "#7dd3fc" if rank == 0 else "#475569"
                    bold  = "700" if rank == 0 else "400"
                    st.markdown(f"""
                    <div style="margin-bottom:.55rem;">
                      <div style="display:flex;justify-content:space-between;margin-bottom:.2rem;">
                        <span style="color:#e2e8f0;font-weight:{bold};">{rank+1}. {ans}</span>
                        <span style="color:{color};font-weight:600;">{conf*100:.1f}%</span>
                      </div>
                      <div style="background:#1e293b;border-radius:4px;height:5px;">
                        <div style="background:{color};width:{pct}%;height:5px;border-radius:4px;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        except Exception as e:
            status.update(label="Inference failed.", state="error", expanded=True)
            st.error(str(e))
            st.exception(e)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#475569;font-size:.78rem;'>"
    "QG-SAM &nbsp;|&nbsp; BERT + CLIP ViT + Evidence-First Reasoning"
    "</p>",
    unsafe_allow_html=True,
)
