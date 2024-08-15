# MobileAIBench

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