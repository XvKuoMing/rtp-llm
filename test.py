from src.rtp_llm.providers.alpaca import OpenAIProvider



provider = OpenAIProvider(
    overwrite_tts_model_api_key="sk-proj-1234567890",
    overwrite_tts_model_base_url="https://api.openai.com/v1",
    tts_model="gpt-4o-mini-tts"
)

print(provider)
print(provider.tts_client)






