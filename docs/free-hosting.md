# Free Streamlit hosting

This repository is ready for free Streamlit-capable hosting through the root
`app.py` entrypoint.

## Option 1: Streamlit Community Cloud

1. Push this branch to GitHub.
2. Go to <https://share.streamlit.io/>.
3. Create a new app from this repository.
4. Set the app file to:

   ```text
   app.py
   ```

5. Add secrets in the Streamlit Cloud app settings:

   ```toml
   OPENAI_API_KEY = "your_openrouter_or_openai_key"
   TAVILY_API_KEY = "your_tavily_key"
   LLM_PROVIDER = "openrouter"
   OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
   OPENAI_MODEL = "openai/gpt-4o-mini"
   ```

6. Deploy.

## Option 2: Hugging Face Spaces

1. Create a new Space at <https://huggingface.co/spaces>.
2. Choose **Streamlit** as the SDK.
3. Upload or connect this repository.
4. Keep the app file as:

   ```text
   app.py
   ```

5. Add repository secrets:

   ```text
   OPENAI_API_KEY
   TAVILY_API_KEY
   LLM_PROVIDER=openrouter
   OPENAI_BASE_URL=https://openrouter.ai/api/v1
   OPENAI_MODEL=openai/gpt-4o-mini
   ```

6. The Space will install `requirements.txt` and launch the Streamlit app.

## Local check

```bash
streamlit run app.py
```

The nested app implementation remains at:

```text
src/deep_researcher/main/streamlit_app.py
```
