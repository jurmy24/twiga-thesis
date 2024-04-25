import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# torch.set_default_device("cuda")
# if this goes to shit make sure I delete this model.safetensors cus its almost 3GB

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", torch_dtype=torch.float32)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")

inputs = tokenizer('''def print_prime(n):
   """
   Print all primes between 1 and n
   """''', return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=200)
text = tokenizer.batch_decode(outputs)[0]
print(text)