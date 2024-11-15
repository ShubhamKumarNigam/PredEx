<h1 align="center">
<img src="Assets/L-NLP_Logo_Transparent.png" width="100" alt="L-NLP" />
<br>
Legal Judgment Reimagined: PredEx and the Rise of Intelligent AI Interpretation in Indian Courts (ACL 2024)
</h1>

![task_desc](https://github.com/ShubhamKumarNigam/PredEx/raw/main/Assets/task_desc.jpg)

<p align="center">
  <a href="https://huggingface.co/L-NLProc"><b>[🌐 Website]</b></a> •
  <a href="https://aclanthology.org/2024.findings-acl.255/"><b>[📜 Proceedings]</b></a> •
  <a href="https://arxiv.org/abs/2406.04136"><b>[📜 ArXiv]</b></a> •
  <a href="https://huggingface.co/collections/L-NLProc/predex-models-66509d3f4de624770d690a48"><b>[🤗 HF Models]</b></a> •
  <a href="https://huggingface.co/collections/L-NLProc/predex-datasets-6650a75907cc2255eab18d01"><b>[🤗 HF Dataset]</b></a> •
  <a href="https://github.com/ShubhamKumarNigam/PredEx"><b>[🐱 GitHub]</b></a>
</p>

<p align="center">
  This is the official implementation of the paper:
</p>

<p align="center">
  <a href="https://sites.google.com/view/shubhamkumarnigam">Shubham Kumar Nigam</a>, <a href="https://www.linkedin.com/in/anuragsharma321/">Anurag Sharma</a>, <a href="https://www.linkedin.com/in/danushk/">Danush Khanna</a>, <a href="#">Noel Shallum</a>, <a href="https://sites.google.com/view/kripabandhughosh-homepage/home">Kripabandhu Ghosh</a>, and <a href="https://www.cse.iitk.ac.in/users/arnabb/">Arnab Bhattacharya</a>:
</p>

<p align="center">
  <a href="https://aclanthology.org/2024.findings-acl.255/">Legal Judgment Reimagined: PredEx and the Rise of Intelligent AI Interpretation in Indian Courts</a> (to appear in <strong>ACL 2024</strong>)
</p>

LLMs, used for legal outcome prediction and explainability, face challenges due to the complexity of legal proceedings and limited expert-annotated data. PredEx tackles this with the largest expert-annotated dataset based on Indian legal documents, featuring over 15,000 annotations. Our best Transformer model, Roberta, achieves 78% accuracy, surpassing LLama-2-7B at 38% and human experts at 73%. PredEx sets a new benchmark for legal judgment prediction in the NLP community!     
See also our [**Linkedin Post**](https://www.linkedin.com/posts/shubham-kumar-nigam-34670541_pdf-activity-7196821209903181825-tki8?utm_source=share&utm_medium=member_desktop).

PredEx can be used to improve the performance of already-trained large language models not only in legal outcome prediction but also in providing meaningful reasoning behind their decisions. For best results, the models can be trained with PredEx.

If you have any questions on this work, please open a [GitHub issue](https://github.com/ShubhamKumarNigam/PredEx/issues) or email the authors at

```shubhamkumarnigam@gmail.com, anuragsharma3211@gmail.com, danush.s.khanna@gmail.com```

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
| Predex | InLegalBert  | [L-NLProc/PredEx_InLegalBert_Pred](https://huggingface.co/L-NLProc/PredEx_InLegalBert_Pred)  |
| Predex | InCaseLaw  | [L-NLProc/PredEx_InCaseLaw_Pred](https://huggingface.co/L-NLProc/PredEx_InCaseLaw_Pred) |
| Predex | XLNet Large  | [L-NLProc/PredEx_XLNet_Large_Pred](https://huggingface.co/L-NLProc/PredEx_XLNet_Large_Pred)  |
| Predex | RoBerta Large  | [L-NLProc/PredEx_RoBERTa_Large_Pred](https://huggingface.co/L-NLProc/PredEx_RoBERTa_Large_Pred)  |

### Table 2: Prediction only, LLM-based models on PredEx
| Dataset |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| Predex | Zephyr  | [Zephyr Hugging Face API](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta)  |
| Predex | Gemini pro  | [Build with Gemini API](https://ai.google.dev/gemini-api?gad_source=1) |
| Predex | Llama-2-7B  | [L-NLProc/PredEx_Llama-2-7B_Pred](https://huggingface.co/L-NLProc/PredEx_Llama-2-7B_Pred)  |
| Predex | Llama-2-7B Instruction-tuning on prediction task  | [L-NLProc/PredEx_Llama-2-7B_Pred_Instruction-Tuned](https://huggingface.co/L-NLProc/PredEx_Llama-2-7B_Pred_Instruction-Tuned)  |

### Table 3: Prediction with the explanation on PredEx, LLM-based models
| Dataset |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| Predex | Gemini pro  | [Build with Gemini API](https://ai.google.dev/gemini-api?gad_source=1)  |
| Predex | Llama-2-7B  | [L-NLProc/PredEx_Llama-2-7B_Pred-Exp](https://huggingface.co/L-NLProc/PredEx_Llama-2-7B_Pred-Exp) |
| Predex | Llama-2-7B Instruction-tuning on prediction with explanation task  | [L-NLProc/PredEx_Llama-2-7B_Pred-Exp_Instruction-Tuned](https://huggingface.co/L-NLProc/PredEx_Llama-2-7B_Pred-Exp_Instruction-Tuned)  |

### Table 4: Prediction with the explanation on ILDC Expert, LLM-based models
| Dataset |  Method | Hugging Face link |
| ------------- | ------------- | ------------- |
| ILDC Expert | Llama-2-7B  | [L-NLProc/ILDC_Llama-2-7B_Pred-Exp](https://huggingface.co/L-NLProc/ILDC_Llama-2-7B_Pred-Exp)  |
| ILDC Expert | Llama-2-7B Instruction-tuning on prediction with explanation task  | [L-NLProc/ILDC_Llama-2-7B_Pred-Exp_Instruction-Tuned](https://huggingface.co/L-NLProc/ILDC_Llama-2-7B_Pred-Exp_Instruction-Tuned) |

## Results

<img width="50%" alt="image" src="https://github.com/ShubhamKumarNigam/PredEx/blob/main/Assets/table3.png">
<img width="80%" alt="image" src="https://github.com/ShubhamKumarNigam/PredEx/blob/main/Assets/table4.png">
<img width="40%" alt="image" src="https://github.com/ShubhamKumarNigam/PredEx/blob/main/Assets/table5.png">


## Citation
If you use our method or models, please cite [our paper](https://aclanthology.org/2024.findings-acl.255/):
```
@inproceedings{nigam-etal-2024-legal,
    title = "Legal Judgment Reimagined: {P}red{E}x and the Rise of Intelligent {AI} Interpretation in {I}ndian Courts",
    author = "Nigam, Shubham  and
      Sharma, Anurag  and
      Khanna, Danush  and
      Shallum, Noel  and
      Ghosh, Kripabandhu  and
      Bhattacharya, Arnab",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.255",
    doi = "10.18653/v1/2024.findings-acl.255",
    pages = "4296--4315",
    abstract = "In the era of Large Language Models (LLMs), predicting judicial outcomes poses significant challenges due to the complexity of legal proceedings and the scarcity of expert-annotated datasets. Addressing this, we introduce \textbf{Pred}iction with \textbf{Ex}planation (PredEx), the largest expert-annotated dataset for legal judgment prediction and explanation in the Indian context, featuring over 15,000 annotations. This groundbreaking corpus significantly enhances the training and evaluation of AI models in legal analysis, with innovations including the application of instruction tuning to LLMs. This method has markedly improved the predictive accuracy and explanatory depth of these models for legal judgments. We employed various transformer-based models, tailored for both general and Indian legal contexts. Through rigorous lexical, semantic, and expert assessments, our models effectively leverage PredEx to provide precise predictions and meaningful explanations, establishing it as a valuable benchmark for both the legal profession and the NLP community.",
}
```



