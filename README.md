<div align="center">
  <a href="https://github.com/SalesforceAIResearch/MobileAIBench"><img width="300px" height="auto" src="./image/logo1.png"></a>
  <h1>MobileAIBench</h1>
</div>

<div align="center">
    
  ![Python 3.10](https://img.shields.io/pypi/pyversions/evogfuzz)
  [![License](https://img.shields.io/badge/License-Apache-green.svg)]()
  
</div>


<p align="center">
  <a href="https://arxiv.org/abs/2406.10290">Paper</a> |
  <a href="https://github.com/SalesforceAIResearch/MobileAIBench?tab=readme-ov-file#installation">Installation</a> |
  <a href="https://github.com/SalesforceAIResearch/MobileAIBench?tab=readme-ov-file#usage">Usage</a> |
  <a href="https://github.com/SalesforceAIResearch/MobileAIBench?tab=readme-ov-file#running-mobile-app">Mobile App</a> 
</p>

---

## MobileAIBench

A comprehensive benchmark designed to evaluate the performance and resource consumptions of LLMs & LMMs for on-device use cases.


## Installation

To install MobileBench, follow these steps:

1. Clone the Repository:
   ```shell
   git clone --recurse-submodules https://github.com/SalesforceAIResearch/MobileAIBench.git
   ```
2. Create a Conda Environment:
   ```shell
   conda create -n mobile_bench python=3.10
   conda activate mobile_bench
   ```
3. Run the Makefile:
   ```shell
   make
   ```
4. Add OpenAI API Key:
   ```shell
   export OPENAI_API_KEY=<OPENAI_API_KEY>
   ```

## Usage

Here are some usage examples for running MobileAIBench:

### Task: Question Answering

- **Dataset:** hotpot_qa & databricks-15k
- **Model:** xgen2-3b.gguf

- Run on GPU:
    ```bash
    python ./src/mobile_bench.py --task question_answering --model_lib llama_cpp_python --model_name xgen2-3b.gguf --use_gpu
    ```
- Run on CPU:
    ```bash
    python ./src/mobile_bench.py --task question_answering --model_lib llama_cpp_python --model_name xgen2-3b.gguf
    ```

### Task: All (Standard_NLP and Trust & Safety)

- **Model:** xgen2-3b.gguf

- Run on GPU:
    ```bash
    python ./src/mobile_bench.py --task all --model_lib llama_cpp_python --model_name xgen2-3b.gguf --use_gpu
    ```
- Run on CPU:
    ```bash
    python ./src/mobile_bench.py --task all --model_lib llama_cpp_python --model_name xgen2-3b.gguf
    ```
## Running Mobile App
- MobileAIBench supports both iOS and Android 
- For iOS
  
## Prerequisites

1. **Apple Developer Account:** Ensure you have signed up for an Apple Developer account(free version).

## Steps to Run the Project

1. **Open the Project in Xcode:**
   - Open `MobileAIBench.xcodeproj` located at `ios-app/MobileAIBench/MobileAIBench.xcodeproj` in Xcode.

2. **Configure Xcode Settings:**
   - Change the Xcode developer settings to your personal team.
   - Change the bundle identifier name to your choice.(Make sure that its in following format: ORGANIZATION_NAME.APP_NAME, if organization doesnt exist then just keep APP_NAME)

3. **Build and Run the App:**
   - Build and run the app on your iPhone.
   - While running on a mobile device, open in developer mode, and trust the developer of the app in the settings.

4. **Add Models to the App Directory:**
   - Copy and paste the following models into the iPhone's app directory, ensuring they are in `.gguf` format and named correctly:
     - `tinyllama-1.1b-chat_Q4_K_M.gguf`
     - `phi-2_Q4_K_M.gguf`
     - `gemma-2b-it_Q4_K_M.gguf`
     - `stablelm-zephyr-3b_Q4_K_M.gguf`
   - Additionally, copy and paste the multimodal model and projector as follows:
     - `llava-phi-2.gguf`
     - `llava-phi-2-mmproj.gguf`

5. **Run the App:**
   - Select your iPhone as the target device.
   - Run the app using the play button in Xcode.
   - After building and running the app on the mobile device using Xcode, close Xcode.

6. **Record Performance Metrics:**
   - To record CPU, RAM, memory, and thermal usage, open the Instrument app.
   - Run the LLMBench project using the Instrument app.
   - Depending on the metrics you want to record, choose profiles such as Activity Monitor, GPU, Metal Application, and Thermal State.
## Download Models

Download all the required models in `.gguf` format from the following link: [Download](https://huggingface.co/tulika214/Quantized_4_bit_models/tree/main)

- For Android:

## Prerequisites

Before you begin, ensure that you have the following:

1. **Android Studio**: Installed on your computer. You can download it from [here](https://developer.android.com/studio).
2. **Android Device**: Optional but recommended for testing on a real device.
3. **Android Developer Account**: Optional, but needed if you want to distribute your app on the Google Play Store.

## Steps to Load and Run the Project

1. **Open the Project in Android Studio:**
   - Open your project by selecting Open an existing project or going to File > Open, Open `MobileAIBench` located at `android-app/MobileAIBench/examples/llama.android` in Android Studio.

2. **Build the App:**
   - When you open a project, Android Studio usually prompts you to sync the Gradle files automatically. If this happens, just click Sync Now in the prompt that appears.
   - If the prompt does not appear, or if you’ve made changes to your build.gradle files, you may need to manually sync. Click on the Sync Project with Gradle Files button (usually represented by an elephant icon) or go to File > Sync Project with Gradle Files.

4. **Run the App:**
   - While running on a mobile device, open in developer mode, and trust the developer of the app in the settings.
   - Select your android as the target device.
   - Run the app using the play button in Android Studio.
   - After building and running the app on the mobile device using Android Studio, close Android Studio.
   - App is now ready to be used on the target device!

## Implementation Basis
- This Android app is based on the llama.android implementation from the official llama.cpp repository. You can find it [here](https://github.com/ggerganov/llama.cpp/tree/master/examples/llama.android).

