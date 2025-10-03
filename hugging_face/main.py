from transformers import pipeline

import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = pipeline("summarization",model = "facebook/bart-large-cnn")

response = model("Ollama is a great tool for running large language models locally on your own hardware. It provides an easy-to-use interface for interacting with these models and allows you to run them without needing an internet connection. This makes it a great option for those who want to use large language models but are concerned about privacy or data security. Additionally, Ollama is designed to be efficient and can run on a variety of hardware configurations, making it accessible to a wide range of users.", max_length=50, min_length=25, do_sample=False)

print(response)