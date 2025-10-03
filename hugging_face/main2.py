from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

model = pipeline("text-generation", model="LiquidAI/LFM2-2.6B", device=0, max_length=256,truncation=True)

llmm = HuggingFacePipeline(pipeline=model)

template = PromptTemplate.from_template("Explain {topic} in detaail for a {age} year old to understand")

chain = template | llmm

topic = input("Topic: ")


age = input("Age: ")

response = chain.invoke({"topic":topic,"age":age})
print(response)