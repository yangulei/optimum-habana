# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="abhinand/llama-2-13b-hf-bf16-sharded", trust_remote_code=True)