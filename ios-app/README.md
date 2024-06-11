# MobileAIBench

## Prerequisites

1. **Apple Developer Account:** Ensure you have signed up for an Apple Developer account(free version).

## Steps to Run the Project

1. **Open the Project in Xcode:**
   - Open `MobileAIBench.xcodeproj` located at `ios-app/MobileAIBench/MobileAIBench.xcodeproj` in Xcode.

2. **Configure Xcode Settings:**
   - Change the Xcode developer settings to your personal team.

3. **Build and Run the App:**
   - Build and run the app on your iPhone.

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

6. **Record Performance Metrics:**
   - To record CPU, RAM, memory, and thermal usage, open the Instrument app.
   - Run the LLMBench project using the Instrument app.

## Download Models

Download all the required models in `.gguf` format from the following link: [Download](https://huggingface.co/tulika214/Quantized_4_bit_models/tree/main)
