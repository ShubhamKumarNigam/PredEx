# Legal Judgment Reimagined: PredEx and the Rise of Intelligent AI Interpretation in Indian Courts (ACL 2024)
![task_desc](https://github.com/ShubhamKumarNigam/PredEx/raw/main/Assets/task_desc.jpg)

This is the official implementation of the paper:

[Shubham Kumar Nigam](https://sites.google.com/view/shubhamkumarnigam), [Anurag Sharma](https://www.linkedin.com/in/anuragsharma321/), [Danush Khanna](https://www.linkedin.com/in/danushk/), [Noel Shallum](), [Kripabandhu Ghosh](https://sites.google.com/view/kripabandhughosh-homepage/home), and [Arnab Bhattacharya](https://www.cse.iitk.ac.in/users/arnabb/) :  

[Legal Judgment Reimagined: PredEx and the Rise of Intelligent AI Interpretation in Indian Courts](https://shorturl.at/qeuWr) (to appear in **ACL 2024**)

LLMs, used for legal outcome prediction and explainability, face challenges due to the complexity of legal proceedings and limited expert-annotated data. PredEx tackles this with the largest expert-annotated dataset based on Indian legal documents, featuring over 15,000 annotations. Our best Transformer model, Roberta, achieves 78% accuracy, surpassing LLama-2-7B at 38% and human experts at 73%. PredEx sets a new benchmark for legal judgment prediction in the NLP community!     
See also our [**Linkedin Post**](https://www.linkedin.com/posts/shubham-kumar-nigam-34670541_pdf-activity-7196821209903181825-tki8?utm_source=share&utm_medium=member_desktop).

PredEx can be used to improve the performance of already-trained large language models not only in legal outcome prediction but also in providing meaningful reasoning behind their decisions. For best results, the models can be trained with PredEx.

If you have any questions on this work, please open a [GitHub issue](https://github.com/ShubhamKumarNigam/PredEx/issues) or email the authors at ```shubhamkumarnigam@gmail.com, anuragsharma3211@gmail.com, danush.s.khanna@gmail.com```

## **May 2024** - PredEx will appear at ACL 2024!

## Getting Started

### General Instructions

Ensure you have the necessary hardware and software requirements in place to replicate our experimental setup. Follow the steps below to configure your environment for optimal performance.

## Recommended Hardware Configuration

### Hardware Specifications

- Utilize two cores of [NVIDIA A100-PCIE-40GB](https://www.nvidia.com/en-gb/data-center/a100/) with 126GB RAM of 32 cores for instruction fine-tuning.
- Additionally, a Google Colab Pro subscription with A100 Hardware accelerator is recommended for conducting inference and other experiments.

## Recommended Software Configuration

### Software Setup

- Set up the environment with appropriate drivers and libraries for GPU acceleration.
- Install necessary dependencies for model training and inference.

## Model Training Specifics

### Fine-tuning Parameters

- Fine-tune the Large Language Models (LLMs) for 5 epochs to achieve a balance between training adequacy and preventing overfitting.

### Post-processing for Quality Enhancement

- Implement a post-processing step after inference to mitigate common issues with generative models, such as sentence hallucination and repetition.
- Select the initial occurrences of decision and explanation parts from the model outputs and omit subsequent repetitions to refine output quality, ensuring coherence and conciseness.

## Evaluation Process

### Handling Non-inferential Results

- Exclude cases where certain LLMs do not yield inference results to maintain the integrity and accuracy of experimental findings.
- By excluding non-inferential results, ensure that the evaluation process remains unbiased and reflective of the models' performance.


## Trained Models
The following models from the paper (Table 3) are available on Hugging Face. 

### Table 1: Prediction only, LM-based models on PredEx
| Dataset |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| Predex | InLegalBert  | [abertsch/bart-base-booksum](https://huggingface.co/abertsch/bart-base-booksum)  |
| Predex | InCaseLaw  | [abertsch/unlimiformer-bart-booksum-earlyk](https://huggingface.co/abertsch/unlimiformer-bart-booksum-earlyk) |
| Predex | XLNet Large  | [abertsch/unlimiformer-bart-booksum-random-encoding](https://huggingface.co/abertsch/unlimiformer-bart-booksum-random-encoding)  |
| Predex | RoBerta Large  | [abertsch/unlimiformer-bart-booksum-alternating](https://huggingface.co/abertsch/unlimiformer-bart-booksum-alternating)  |

### Table 2: Prediction only, LLM-based models on PredEx
| Dataset |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| Predex | Zephyr  | [abertsch/bart-base-booksum](https://huggingface.co/abertsch/bart-base-booksum)  |
| Predex | Gemini pro  | [abertsch/unlimiformer-bart-booksum-earlyk](https://huggingface.co/abertsch/unlimiformer-bart-booksum-earlyk) |
| Predex | Llama-2-7B  | [abertsch/unlimiformer-bart-booksum-random-encoding](https://huggingface.co/abertsch/unlimiformer-bart-booksum-random-encoding)  |
| Predex | Llama-2-7B Instruction-tuning on prediction with explanation task  | [abertsch/unlimiformer-bart-booksum-alternating](https://huggingface.co/abertsch/unlimiformer-bart-booksum-alternating)  |

### Table 3: Prediction with the explanation on PredEx, LLM-based models
| Dataset |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| Predex | Gemini pro  | [abertsch/bart-base-booksum](https://huggingface.co/abertsch/bart-base-booksum)  |
| Predex | Llama-2-7B  | [abertsch/unlimiformer-bart-booksum-earlyk](https://huggingface.co/abertsch/unlimiformer-bart-booksum-earlyk) |
| Predex | Llama-2-7B Instruction-tuning on prediction with explanation task  | [abertsch/unlimiformer-bart-booksum-random-encoding](https://huggingface.co/abertsch/unlimiformer-bart-booksum-random-encoding)  |

### Table 4: Prediction with the explanation on ILDC Expert, LLM-based models
| Dataset |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| ILDC Expert | Llama-2-7B  | [abertsch/bart-base-booksum](https://huggingface.co/abertsch/bart-base-booksum)  |
| ILDC Expert | Llama-2-7B Instruction-tuning on prediction with explanation task  | [abertsch/unlimiformer-bart-booksum-earlyk](https://huggingface.co/abertsch/unlimiformer-bart-booksum-earlyk) |

## Results

<img width="50%" alt="image" src="https://github.com/ShubhamKumarNigam/PredEx/blob/main/Assets/table3.png">
<img width="75%" alt="image" src="https://github.com/ShubhamKumarNigam/PredEx/blob/main/Assets/table4.png">
<img width="50%" alt="image" src="https://github.com/ShubhamKumarNigam/PredEx/blob/main/Assets/table5.png">


## Citation
If you use our method or models, please cite [our paper](https://shorturl.at/qeuWr):
```
@inproceedings{
anonymous2024legal,
title={Legal Judgment Reimagined: PredEx and the Rise of Intelligent {AI} Interpretation in Indian Courts},
author={Anonymous},
booktitle={The 62nd Annual Meeting of the Association for Computational Linguistics},
year={2024},
url={https://openreview.net/forum?id=aZIwY6nOBq}
}  
```



