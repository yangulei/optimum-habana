# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="internlm/internlm-20b", trust_remote_code=True)