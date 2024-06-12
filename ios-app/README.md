# MobileAIBench

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
