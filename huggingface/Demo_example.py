from transformers.pipelines import pipeline
import torch

# Define the models
model_llama = "meta-llama/Llama-2-13b"
model_danube = "h2oai/h2o-danube-1.8b-chat"
model_gemma = "google/gemma-2-2b-it"

# Define the input messages
messages = [
    [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "Write a poem on Hugging Face, the company"}],
        },
    ],
]

# Create pipelines once (outside of loop)
pipe_llama = pipeline(
    "text-generation",
    model=model_llama,
    device="cuda",
    torch_dtype=torch.bfloat16,
)

pipe_danube = pipeline(
    "text-generation",
    model=model_danube,
    device="cuda",
    torch_dtype=torch.bfloat16,
)

pipe_gemma = pipeline(
    "text-generation",
    model=model_gemma,
    device="cuda",
    torch_dtype=torch.bfloat16,
)

# Run all pipelines
print("\n--- LLaMA 2 13B ---\n")
print(pipe_llama(messages, max_new_tokens=50))

print("\n--- H2O Danube 1.8B Chat ---\n")
print(pipe_danube(messages, max_new_tokens=50))

print("\n--- Google Gemma 2 2B IT ---\n")
print(pipe_gemma(messages, max_new_tokens=50))
