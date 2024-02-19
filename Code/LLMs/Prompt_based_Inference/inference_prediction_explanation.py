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

peft_model_dir = "llama_pred_exp" #finetuned model dir
# load base LLM model and tokenizer
trained_model = AutoPeftModelForCausalLM.from_pretrained(
    peft_model_dir,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(peft_model_dir)

df["Prediction"] = ""
for i, row in tqdm(df.iterrows()):
    case_pro = row["Input"]
    prompt = f"""Analyze the case proceeding and predict whether the appeal/petition will be accepted (1) or rejected (0). \
  and subsequently provide an Explanation behind this prediction with important textual evidence from the case.
  
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
    outputs = trained_model.generate(input_ids=input_ids, max_new_tokens=512, )
    output = tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0][len(prompt):]
    df.at[i,"Prediction"] = output

df.head()
df.to_csv("results/pred_exp.csv", index = False)

