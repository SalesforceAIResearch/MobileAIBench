# MobileAIBench-Multimodal
Evaluation for multi-modal LLMs with different level of quantization using Llama.cpp

## Setup
### Create a conda environment
Use the following commands to create and activate a new Conda environment.
```shell
conda create -n ${YOUR_ENV_NAME} python=3.11  # tested with 3.11, other versions may also work.
conda activate ${YOUR_ENV_NAME}
```
### Install llama-cpp-python
```shell
pip install llama-cpp-python
```

## Instruction for testing multi-modal models on VQA tasks.
### 1. After quantizing the models, setup the model path to the GGUF files in multimodal/config/model_config.yaml. Example for llava-v1.5-7b under 8-bit quantization:
```shell
llava-v1.5-7b-q8-0:
  name: Llava-v1.5-7b
  model_path: .../llava-v1.5-7b-q8-0.gguf
  chat_handler_path: .../llava-v1.5-7b-mmproj.gguf
  size: 7B
  context_length: 4096
```
### 2. Download the datasets and setup the root dataset directory in multimodal/evaluate_vqa.sh. ScienceQA and TextVQA are already available on HuggingFace and the link are integrated, therefore, experiments on these two datasets can be directly tested without manually downloading them.
### 3. For testing a single model on a single dataset, simply run:
```shell
    sh multimodal/scripts/evaluate_vqa.sh llava-v1.5-7b-q8-0 vqav2
```
### 4. For testing multiple models on multiple datasets:
```shell
    bash multimodal/scripts/evaluate_all.sh
```
### 5. Aftering finish testing, the logs will be created in /logs folder.

## Open source VQA Dataset Links. (ScienceQA and TextVQA datasets are already integrated into the code, experiments on these two datasets can be directly tested without manually downloading them). 
### [VQA-v2](https://visualqa.org/index.html); [Viswiz](https://vizwiz.org/tasks-and-datasets/vqa/); [GQA](https://cs.stanford.edu/people/dorarad/gqa/index.html)