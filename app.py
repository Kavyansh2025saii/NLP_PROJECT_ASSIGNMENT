"""TweetGuard AI: four-level social content risk advisor."""
import html
import os
import re
from datetime import datetime

import pandas as pd
import streamlit as st

st.set_page_config(page_title="TweetGuard AI | Pre-post intelligence", page_icon="🛡️", layout="wide", initial_sidebar_state="collapsed")

COLORS = {"Green": "#36d399", "Yellow": "#f7c95b", "Orange": "#ff944d", "Red": "#ff6178"}
LEVELS = ["Green", "Yellow", "Orange", "Red"]
DESCRIPTIONS = {
    "Green": "No obvious communication-risk indicators detected.",
    "Yellow": "Potentially inflammatory or ambiguous wording. Context may help.",
    "Orange": "Potentially discriminatory or exclusionary language.",
    "Red": "Possible direct encouragement of violence or severe targeted hostility.",
}

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*=css],.stApp {font-family:'DM Sans',sans-serif;}
.stApp {background: radial-gradient(ellipse at 80% 0%,#192546 0%,transparent 39%),linear-gradient(135deg,#080e1b,#101728 63%,#0b1020); color:#e9edfa;}
.block-container {max-width:1160px;padding-top:2.0rem;padding-bottom:3.6rem;}
#MainMenu,footer,header {visibility:hidden;}
h1,h2,h3 {font-family:'Space Grotesk',sans-serif!important;letter-spacing:-.03em!important;}
p {color:#b6c1d9;}
.hero-title {font-family:'Space Grotesk',sans-serif;font-size:clamp(2.5rem,5.4vw,4.5rem);font-weight:700;line-height:1.06;letter-spacing:-.06em;color:#f9fbff;margin:.45rem 0 .55rem;}
.hero-title span {color:#66e5ca;}
.eyebrow {font-size:.75rem;letter-spacing:.2em;color:#70e6d0;font-weight:800;}
.hero-sub {max-width:650px;font-size:1rem;line-height:1.7;color:#aab7d4;margin-bottom:1.6rem;}
.glass {background:rgba(19,29,50,.75);border:1px solid #263754;border-radius:18px;padding:20px 22px;margin:4px 0 18px;}
.kicker {font-size:.73rem;font-weight:800;letter-spacing:.15em;color:#94a7ca;text-transform:uppercase;}
.pill {display:inline-block;padding:7px 12px;border-radius:100px;border:1px solid #31405f;background:#15213a;font-size:.72rem;color:#b9c8e5;font-weight:800;}
.result-title {font-size:1.9rem;font-family:'Space Grotesk',sans-serif;font-weight:700;margin:.45rem 0;}
.risk-track {background:#26314a;height:11px;border-radius:20px;overflow:hidden;margin:16px 0 7px;}
.risk-progress {height:100%;border-radius:20px;}
.section-head {font-family:'Space Grotesk',sans-serif;font-size:1.25rem;font-weight:700;color:#e9effe;margin:1.2rem 0 .8rem;}
.stTextArea textarea {background:#111e34!important;border:1px solid #334769!important;border-radius:15px!important;color:#eef4ff!important;font-size:1.03rem!important;line-height:1.6!important;}
.stTextArea textarea:focus {border-color:#63dfc5!important;box-shadow:0 0 0 1px #63dfc5!important;}
.stButton button[kind=primary] {background:linear-gradient(90deg,#5de1c4,#4bbacb)!important;color:#061526!important;border:none!important;font-weight:800!important;border-radius:11px!important;}
.stButton button {border-radius:11px!important;font-weight:700!important;}
div[data-testid=stMetric] {background:#121e33;border:1px solid #293958;border-radius:13px;padding:15px;}
div[data-testid=stMetricValue] {color:#f2f6ff;}
div[data-testid=stAlert] {border-radius:12px;}
[data-testid=stVerticalBlockBorderWrapper] {border-color:#2b3b58!important;}
</style>""", unsafe_allow_html=True)


def escape(text):
    return html.escape(str(text), quote=True)


def rule_analyze(text):
    """Deterministic baseline, deliberately not a trained ML predictor."""
    t = text.lower()
    quoted = bool(re.search(r'\b(quoted|someone said|they said|i disagree|i condemn|wrong to say|reported)\b', t))
    violence = bool(re.search(r'\b(kill|attack|beat up|hurt|shoot)\b', t))
    urging = bool(re.search(r'\b(let.s|we should|go|must|should|i will|i.m going to)\b', t))
    targeted = bool(re.search(r'\b(women|men|girls|boys|athletes|people|them|those|players|immigrants)\b', t))
    exclusion = bool(re.search(r'\b(women|men|girls|boys)\b.{0,65}\b(weak|inferior|should not|shouldn.t|cannot|can.t|don.t belong|too weak)\b|\b(women|men|girls|boys) should(n.t| not)\b', t))
    hostility = bool(re.search(r'\b(stupid|hate|terrible|idiot|disgusting|ban|worst|dangerous)\b', t))
    concerns = []
    if quoted:
        concerns.append("The text may be reporting or criticizing someone else's words; verify authorship and intent.")
    if violence and urging and targeted and not quoted:
        level = "Red"
        concerns.append("Wording appears to encourage harming people.")
    elif exclusion and not quoted:
        level = "Orange"
        concerns.append("A gender-based generalization or exclusion appears in the text.")
    elif violence or hostility or exclusion:
        level = "Yellow"
        concerns.append("Strong or potentially harmful wording may need more context.")
    else:
        level = "Green"
        concerns.append("The simple rules did not find a known risk pattern.")
    return level, concerns


@st.cache_resource(show_spinner="Loading the transformer model...")
def get_classifier():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)


def transformer_analyze(text):
    labels = [
        "ordinary constructive or harmless social media post",
        "controversial or ambiguous social media opinion",
        "discriminatory stereotype or exclusionary statement",
        "explicit incitement to violence or severe targeted abuse",
    ]
    classifier = get_classifier()
    out = classifier(text, candidate_labels=labels, multi_label=False, hypothesis_template="This post is {}.")
    scores = {LEVELS[labels.index(label)]: float(score) for label, score in zip(out["labels"], out["scores"])}
    level = max(scores, key=scores.get)
    return level, ["Transformer zero-shot category match. Review the tweet's intent and context before relying on the result."], scores


def fallback_rewrite(level, text):
    if level == "Green":
        return "No rewrite needed. Your original wording can be kept."
    if level == "Yellow":
        return "Try this structure: ‘I disagree with [specific policy or claim] because [your evidence]. I think [constructive alternative].’"
    if level == "Orange":
        if re.search(r"\bwomen\b.*\b(sport|weak|athlet)", text.lower()):
            return "Sports should be accessible to everyone, with individual fitness, training and safety considerations guiding participation."
        return "Describe the specific behavior or policy you disagree with, rather than making a claim about an entire group."
    return "Remove any call for harm. State your concern or disagreement peacefully and specifically, without threats."


def generate_llm_rewrite(text, level):
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=os.getenv("TWEETGUARD_LLM_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": "You are a careful social media writing assistant. Suggest ONE succinct rewrite (maximum 280 characters) that preserves the user's defensible intended point while avoiding discriminatory generalizations, threats, and abuse. Do not reframe discriminatory assumptions as facts. If the intention is unclear, ask a single short clarification question instead of guessing. Return only the rewrite or question."},
            {"role": "user", "content": f"Assessed risk: {level}\nOriginal post:\n{text}"},
        ],
        temperature=0.3,
        max_tokens=130,
    )
    return response.choices[0].message.content.strip()


if "tweet" not in st.session_state:
    st.session_state.tweet = "Women should not participate in sports because they are physically weak."
if "result" not in st.session_state:
    st.session_state.result = None
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown('<div class="eyebrow">● YOUR PRE-POST INTELLIGENCE SYSTEM</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">Think before you <span>post.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">An intelligent second opinion for your social media content. Understand communication risks, explore four levels of caution, and refine your message without losing your voice.</div>', unsafe_allow_html=True)
st.markdown('<span class="pill">✦ TWEETGUARD AI</span> &nbsp; <span class="pill">4-TIER RISK ANALYSIS</span> &nbsp; <span class="pill">REWRITE STUDIO</span>', unsafe_allow_html=True)
st.write("")

left, right = st.columns([1.35, 1], gap="large")
with left:
    st.markdown('<div class="section-head">01 / Compose your post</div>', unsafe_allow_html=True)
    st.text_area("Draft tweet", key="tweet", height=180, label_visibility="collapsed", placeholder="What's on your mind?")
    st.caption(f"{len(st.session_state.tweet)} characters · Supports longer drafts for analysis")
    backend = st.selectbox("Risk detection engine", ["Rules demo (instant)", "Transformer (zero-shot, downloads model)"], help="The rule-based option is an offline prototype. The transformer performs zero-shot inference; it is not trained on the final 4 risk labels.")
    use_llm = st.toggle("LLM rewrite (requires OPENAI_API_KEY)", value=False)
    if use_llm and not os.getenv("OPENAI_API_KEY"):
        st.info("No OPENAI_API_KEY configured. A transparent template suggestion will be shown instead.")
    b1, b2 = st.columns([2.1, 1])
    with b1:
        analyze_clicked = st.button("✦ Analyze my tweet", type="primary", use_container_width=True)
    with b2:
        if st.button("Reset", use_container_width=True):
            st.session_state.tweet = ""
            st.session_state.result = None
            st.rerun()

    st.markdown('<div class="section-head">Quick examples</div>', unsafe_allow_html=True)
    examples = {
        "🟢 Supportive": "Everyone deserves equal access to sports and training.",
        "🟡 Critical": "This sports policy is terrible and needs to change.",
        "🟠 Stereotype": "Women should not participate in sports because they are physically weak.",
        "🔴 Threatening": "We should attack those players after the match.",
    }
    ex_cols = st.columns(2)
    for i, (label, example) in enumerate(examples.items()):
        with ex_cols[i % 2]:
            if st.button(label, key=f"example_{i}", use_container_width=True):
                st.session_state.tweet = example
                st.session_state.result = None
                st.rerun()

    st.markdown('<div class="section-head">How risk is interpreted</div>', unsafe_allow_html=True)
    for name, desc in DESCRIPTIONS.items():
        st.markdown(f'<div style="padding:10px 0;border-bottom:1px solid #273752;display:flex;gap:12px;align-items:baseline"><span style="font-weight:800;color:{COLORS[name]};min-width:75px">● {name}</span><span style="color:#afbeda;font-size:.85rem">{escape(desc)}</span></div>', unsafe_allow_html=True)

if analyze_clicked:
    text = st.session_state.tweet.strip()
    if not text:
        st.warning("Write a post first to analyze it.")
    else:
        with st.spinner("Checking language and context..."):
            try:
                if backend.startswith("Transformer"):
                    level, concerns, scores = transformer_analyze(text)
                    mode = "Transformer zero-shot"
                else:
                    level, concerns = rule_analyze(text)
                    scores = None
                    mode = "Rule-based demo"
            except Exception as error:
                st.error(f"Transformer couldn't load or run: {error}. Switch to Rules demo to keep working.")
                level, concerns, scores, mode = None, [], None, None
            if level:
                rewrite = fallback_rewrite(level, text)
                rewrite_source = "Template suggestion"
                if use_llm and os.getenv("OPENAI_API_KEY") and level != "Green":
                    try:
                        rewrite = generate_llm_rewrite(text, level)
                        rewrite_source = "LLM suggestion"
                    except Exception as error:
                        st.warning(f"LLM rewrite unavailable; using a template instead: {error}")
                result = {"tweet": text, "level": level, "concerns": concerns, "scores": scores, "rewrite": rewrite, "rewrite_source": rewrite_source, "mode": mode}
                st.session_state.result = result
                st.session_state.history.insert(0, {"Time": datetime.now().strftime("%H:%M:%S"), "Risk": level, "Text": text[:100], "Engine": mode})

with right:
    st.markdown('<div class="section-head">02 / Risk intelligence</div>', unsafe_allow_html=True)
    r = st.session_state.result
    if r is None:
        st.markdown('<div class="glass" style="text-align:center;padding:68px 22px"><div style="font-size:3rem">◎</div><div class="result-title">Ready when you are</div><p>Enter a post and press <b>Analyze my tweet</b> to see the risk assessment here.</p></div>', unsafe_allow_html=True)
    else:
        color = COLORS[r["level"]]
        idx = LEVELS.index(r["level"]) + 1
        st.markdown(f'''<div class="glass" style="border-color:{color}66">
<div class="kicker">RISK ASSESSMENT · {escape(r['mode'])}</div>
<div class="result-title" style="color:{color}">● {escape(r['level'])} <span style="font-size:1rem;color:#a5b8d7;font-weight:500">/ LEVEL {idx} OF 4</span></div>
<div style="color:#d6def0;font-size:.94rem">{escape(DESCRIPTIONS[r['level']])}</div>
<div class="risk-track"><div class="risk-progress" style="width:{idx*25}%;background:{color}"></div></div>
<div style="font-size:.75rem;color:#92a4c5">Advisory classification · Not a guarantee of safety</div></div>''', unsafe_allow_html=True)
        st.markdown('<div class="kicker" style="margin:16px 0 9px">WHY IT WAS FLAGGED</div>', unsafe_allow_html=True)
        for concern in r["concerns"]:
            st.markdown(f'<div class="glass" style="padding:13px 15px;margin-bottom:7px;font-size:.88rem">✦ {escape(concern)}</div>', unsafe_allow_html=True)
        if r["scores"]:
            st.caption("Zero-shot model label-match scores — not calibrated danger probabilities")
            for name in LEVELS:
                value = r["scores"].get(name, 0)
                st.markdown(f'<div style="display:flex;justify-content:space-between;font-size:.8rem;color:#c4d1e8"><span>{name}</span><span>{value*100:.1f}%</span></div><div class="risk-track" style="height:6px;margin:5px 0 10px"><div class="risk-progress" style="width:{value*100:.1f}%;background:{COLORS[name]}"></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-head">03 / Rewrite studio</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="glass"><div class="kicker">{escape(r["rewrite_source"])}</div><div style="color:#eaf5ff;line-height:1.65;margin-top:9px">{escape(r["rewrite"])}</div></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("↗ Use suggested text", use_container_width=True):
                st.session_state.tweet = r["rewrite"]
                st.session_state.result = None
                st.rerun()
        with c2:
            st.download_button("↓ Download review", data=f"Original: {r['tweet']}\nRisk: {r['level']}\nEngine: {r['mode']}\nReason: {'; '.join(r['concerns'])}\nSuggestion ({r['rewrite_source']}): {r['rewrite']}\n", file_name="tweetguard_review.txt", mime="text/plain", use_container_width=True)
        st.caption("The writer makes the final decision. A low-risk label is not approval or verification.")

if st.session_state.history:
    st.markdown('<div class="section-head">Session activity</div>', unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True, hide_index=True)
    st.download_button("Export session CSV", pd.DataFrame(st.session_state.history).to_csv(index=False).encode(), file_name="tweetguard_session.csv", mime="text/csv")

st.markdown('<div style="padding:32px 0 0;text-align:center;color:#7589a9;font-size:.8rem">TWEETGUARD AI · Built for clearer communication, not censorship · Prototype</div>', unsafe_allow_html=True)
