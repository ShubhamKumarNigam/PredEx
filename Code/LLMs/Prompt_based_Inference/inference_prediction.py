from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
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

peft_model_dir = "llama_pred2" #finetuned model dir

trained_model = AutoPeftModelForCausalLM.from_pretrained(peft_model_dir,
        torch_dtype=torch.float32,
        device_map='auto'
        )
tokenizer = AutoTokenizer.from_pretrained(peft_model_dir)

df["llamaft_pred"] = ""
for i, row in tqdm(df.iterrows()):
    case_pro = row["Input"]
    prompt = f""" ### Instructions:
    Analyze the case proceeding and predict whether the appeal/petition will be rejected (0) or accepted (1). \
  
  ### Input:
  case_proceeding: <{case_pro}>

  ### Response:
  """


    input_ids = tokenizer(prompt, return_tensors='pt',truncation=True).input_ids.cuda()
    # output = tokenizer.decode(
    #     trained_model.generate(
    #         inputs_ids,
    #         max_new_tokens=100,
    #     )[0],
    #     skip_special_tokens=True
    # )
    outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=2, )
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    df.at[i,"llamaft_pred"] = output

df.to_csv("results/llama2_test_pred.csv", index = False)


