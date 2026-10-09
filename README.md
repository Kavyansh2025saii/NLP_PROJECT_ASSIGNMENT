# TweetGuard AI
A polished Streamlit prototype for four-level social-post risk review, optional transformer-based zero-shot classification, and optional LLM rewrite suggestions.

## Start
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
Select **Rules demo** for an instant, offline prototype. Select **Transformer (zero-shot)** to load `facebook/bart-large-mnli` on first use (internet and model download required; can be slow on CPU). It is **not** trained for the four labels, and scores are *not calibrated safety probabilities*.

For genuine LLM rewrite suggestions, set an environment variable `OPENAI_API_KEY` and select **LLM rewrite**. Otherwise the app shows transparent, deterministic rewrite prompts; it does not secretly pass them off as LLM output.

Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your-key"
streamlit run app.py
```

All outputs are advisory, not a content moderation guarantee. Labels measure posting/communication risk, not overall sentiment. Critical opinions and reports/quotes of offensive speech are not automatically endorsements. Test extensively before any deployment.
