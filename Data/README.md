### Train and test set:
Train set: 12178 cases
Test set: 3044 cases

| Case Name | Input | Output | Label | Count | Decision Count |
|----------|----------|----------|----------|----------|----------|
| Petitioner vs. Respondent | Case Text | Case Decision[ds]Explanation | Case Decision | Tokens in case text | Tokens in Explanation text |

Case Decision: 0/1

### Train
* [train.csv](https://drive.google.com/file/d/1eBuQuul8alMDakQTC2TRKN_WzbNQlON3/view?usp=sharing)
### Test
* [test.csv](https://drive.google.com/file/d/1COs3uBBgYz4O09LNL1Slnylxeei02ekp/view?usp=sharing)

### Instructions for Instruction Fine-tuning:
| Instructions	 | Instructions_Exp |
|----------|----------|
| Analyze the case proceeding and predict whether the appeal/petition will be accepted (1) or rejected (0). | First, predict whether the appeal in case proceeding will be accepted (1) or not (0), and then explain the decision by identifying crucial sentences from the document. |

* [instruction_sets.csv](https://drive.google.com/file/d/1YfFzL-0NgFvHWmvlz_vRVrjkw1SwF-dL/view?usp=sharing)

### Fine-tuning data:
Training set: 10961 cases (90% of Train set) <br /> 
Validation set: 1217 cases (10% of Train set)
| Case Name | Input | Output | Label | Count | Decision Count | text |
|----------|----------|----------|----------|----------|----------|----------|
| Petitioner vs. Respondent | Case Text | Case Decision[ds]Explanation | Case Decision | Tokens in case text | Tokens in Explanation text | Prompt for fine-tuning |

Prompt for prediction:
```
Here is an instruction that describes a task, paired with an input (Case text) that provides further context. The case decision is given.

### Instruction:
{instruction}

### Input:
{case_text}

### Response:
{case_decision}
```

 
Prompt for prediction and explanation:
```
Here is an instruction that describes a task, paired with an input (Case text) that provides further context. The case decision and explanation for the decision are given.

### Instruction:
{instruction}

### Input:
{case_text}

### Response:
{case_decision}

### Explanation:
{explanation}
```

* val_1217
  * [LLAMA-2-7B_prediction_with_1000_words_input.csv](https://drive.google.com/file/d/1qrwrTMV5HVKvYmkyG5AMCY5owjbFZxsw/view?usp=sharing)
  * [LLAMA-2-7B_prediction_explanation_with_1000_words_input_1000_words_explanation.csv](https://drive.google.com/file/d/12w-jyO9cASUk8R2C1B3H0799SctJ4p4z/view?usp=sharing)
* train_10961
  * [LLAMA-2-7B_prediction_with_1000_words_input.csv](https://drive.google.com/file/d/1CHj80JoHZEew-OparZ3BnUjfyUdjOlkq/view?usp=sharing)
  * [LLAMA-2-7B_prediction_explanation_with_1000_words_input_1000_words_explanation.csv](https://drive.google.com/file/d/1WQMivt5DpAHegWZDTnXZ4qffJPUmbE7n/view?usp=sharing)
