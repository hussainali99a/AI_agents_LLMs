import ollama 


client = ollama.Client()

model = "gemma3:1b"
prompt = "what is python? explain in detail."

response = client.generate(model=model, prompt=prompt)

print("Ollama response is: ")
print(response.response)
