from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
from tqdm import tqdm

df = pd.read_csv("test.csv")

def preprocess_case(text):
    max_tokens = 1000
    tokens = text.split(' ')
    num_tokens_to_extract = min(max_tokens, len(tokens))
    text1 = ' '.join(tokens[-num_tokens_to_extract:len(tokens)])
    return text1

for i,row in tqdm(df.iterrows()):
    input = row['Input']
    input = preprocess_case(input)
    df.at[i,'Input'] = input

model_id = "meta-llama/Llama-2-7b-chat-hf"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", token= ".....")
tokenizer = AutoTokenizer.from_pretrained(model_id)

def create_prompt(text):
    prompt = f""" ### Instructions:
    Analyze the case proceeding and predict whether the appeal/petition will be rejected (0) or accepted (1). \
  
  ### Input:
  case_proceeding: <{case_pro}>

  ### Response:
  """

    return prompt

df["llama_p"] = ""
for i, row in tqdm(df.iterrows()):
    case_pro = row["Input"]
    prompt = create_prompt(case_pro)
    input_ids = tokenizer(prompt, return_tensors='pt',truncation=True).input_ids.cuda()
    outputs = model.generate(input_ids=input_ids, max_new_tokens=100,)
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    df.at[i,"llama_p"] = output
df.to_csv("llama2_test_pred.csv", index = False)
