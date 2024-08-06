from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = FastAPI()

model_path = './fine-tuned-gpt2'
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token':'[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

class TextRequest(BaseModel):
    prompt: str

def generate_response(prompt, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=max_length,pad_token_id = tokenizer.eos_token_id)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

@app.post("/generate")
async def generate(request: TextRequest):
    response = generate_response(request.prompt)
    return {"response":response}