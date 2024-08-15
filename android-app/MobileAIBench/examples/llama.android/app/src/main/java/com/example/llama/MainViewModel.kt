package com.example.llama

import android.llama.cpp.LLamaAndroid
import android.net.Uri
import android.os.Handler
import android.os.Looper
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.launch
import okhttp3.Call
import okhttp3.Callback
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlin.system.measureTimeMillis


class MainViewModel(private val llamaAndroid: LLamaAndroid = LLamaAndroid.instance()): ViewModel() {
    companion object {
        @JvmStatic
        private val NanosPerSecond = 1_000_000_000.0
    }

    private val tag: String? = this::class.simpleName

    var messageLog by mutableStateOf("")
        private set

    suspend fun send(prompt: String): String {
        messageLog += "\nPredicted Output: "
        var output = ""
        try {
            llamaAndroid.send(prompt)
                .catch { exception ->
                    Log.e(tag, "send() failed", exception)
                    messageLog += exception.message!!
                }
                .collect { value ->
                    println("In: $value")
                    messageLog += value
                    output += value
                }
        } catch (e: Exception) {
            Log.e(tag, "Exception occurred", e)
            messageLog += e.message!!
        }
        return output
    }

    suspend fun load(pathToModel: String) {
        try{
            llamaAndroid.load(pathToModel)
            messageLog += "\nLoaded $pathToModel"
        } catch (exc: IllegalStateException){
            Log.e(tag, "\nload() failed", exc)
            messageLog += exc.message!!
        }
    }
    fun submit_data(name: String, taskName: String, examples: Int, extFilesDir: File){
        println("In bench")
        messageLog += "\nBenchmarking $name for task $taskName with $examples examples\n"
        viewModelScope.launch{
            load_model(name = name, task=taskName, examples=examples, extFilesDir=extFilesDir)
        }

    }
    fun prompt_formation(model: String, question: String, task: String): String {
        val prompt_dict = mutableMapOf<String, String>()
        prompt_dict["phi-2"] = "{system}\nInstruct:{prompt}\nOutput:"
        prompt_dict["TinyLlama-1.1B-Chat"] = "<|system|>\n{system}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>"
        prompt_dict["gemma-2b-it"] = "<start_of_turn>user\n{system}\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
        var prompt = prompt_dict[model]
        var SYS=""
        if (task == "hotpot_qa" || task == "databricks") {
            SYS = "You're a helpful assistant proficient in answering questions. The following context was used to answer the question: {context}"
        } else if (task == "sql_create_context") {
            SYS = "You're a helpful assistant proficient in crafting SQL queries. The following command was used to create the table: {context}\n"
        } else {
            SYS = "You're a helpful assistant in summarizing articles.\n"
        }
        val pprompt = prompt?.replace("{system}", SYS)
        val new_prompt = pprompt?.replace("{prompt}", question)
        if (new_prompt == null) {
            return question
        } else {
            return new_prompt
        }
    }
    suspend fun load_model(name: String, task: String, examples: Int, extFilesDir:File){
        val model_destination = File(extFilesDir, name+".gguf")
        val task_file = File(extFilesDir, task+".json")
        println(model_destination)
        println(task_file)
        if (model_destination.exists()){
            println("model already exists")
            val load_time = measureTimeMillis{ load(model_destination.path) } / 1000.0
            if (task_file.exists()){
                println("task file already exists")
                val jsonString = task_file.readText()
                val jsonArray = Json.parseToJsonElement(jsonString).jsonArray
                var total_time = 0.0
                var output_length=0.0
                for (i in 0 until minOf(examples, jsonArray.size)) {
                    val jsonObject = jsonArray[i].jsonObject
                    val question = jsonObject["question"]?.jsonPrimitive?.content
                    val answer = jsonObject["answer"]?.jsonPrimitive?.content
                    val prompt = question?.let { prompt_formation(name, it, task) }
                    println(prompt)
                    var actual_output=""
                    if (question != null) {
                        if (prompt != null) {
                            messageLog += "\n\nInput: "+question
                            messageLog += "\nActual Output: "+answer
                            total_time += measureTimeMillis{
                                actual_output = send(prompt=prompt) }
                        }
                        val wordCount = getWordCount(actual_output)
                        output_length+=wordCount
                    }
                    println("Question: $question")
                    println("Answer: $answer")
                }
                total_time = total_time / 1000.0
                val avg_time = (total_time)/examples
                val avg_tokens = (output_length)/examples
                val token_per_sec = output_length/total_time
                messageLog += "\n\nDone with benchmarking"
                messageLog += "\nModel load time: "+load_time.toString()+" sec"
                messageLog += "\nTotal time: "+total_time.toString()+" ms"
                messageLog += "\nAverage time: "+avg_time.toString()+" ms"
                messageLog += "\nAverage Number of output tokens: "+avg_tokens.toString()
                messageLog += "\nAverage token per sec: "+token_per_sec.toString()
            }
            else{
                println("task file doesnt exists, downloading file")
                download_task_file(task=task, task_file = task_file)
            }
        }
        else{
            println("model doesnt exists, downloading model")
            download_model(name=name, model_des = model_destination)
        }

    }
    fun getWordCount(sentence: String): Int {
        // Split the sentence by whitespace and filter out any empty strings
        return sentence.split("\\s+".toRegex()).filter { it.isNotEmpty() }.size
    }
    fun download_model(name: String, model_des: File){
        val modelpath = mutableMapOf<String, String>()
        modelpath["phi-2"] = "https://huggingface.co/tulika214/Quantized_4_bit_models/resolve/main/phi-2_Q4_K_M.gguf?download=true"
        modelpath["TinyLlama-1.1B-Chat"] = "https://huggingface.co/tulika214/Quantized_4_bit_models/resolve/main/tinyllama-1.1b-chat_Q4_K_M.gguf?download=true"
        modelpath["gemma-2b-it"] = "https://huggingface.co/tulika214/Quantized_4_bit_models/resolve/main/gemma-2b-it_Q4_K_M.gguf?download=true"
        val model_url = Uri.parse(modelpath[name])
        val mainHandler = Handler(Looper.getMainLooper())
        downloadFile(model_url, model_des) { success, message ->
            mainHandler.post {
                if (success) {
                    messageLog += "\nModel downloaded, run the benchmark again"
                } else {
                    messageLog += "\nModel downloading"
                }
            }
        }
    }
    fun download_task_file(task: String, task_file: File){
        val taskPath = mutableMapOf<String, String>()
        taskPath["hotpot_qa"] = "https://huggingface.co/tulika214/Quantized_4_bit_models/resolve/main/hotpot_qa.json?download=true"
        taskPath["databricks_dolly"] = "https://huggingface.co/tulika214/Quantized_4_bit_models/resolve/main/databricks_dolly.json?download=true"
        taskPath["sql_create_context"] = "https://huggingface.co/tulika214/Quantized_4_bit_models/resolve/main/sql_create_context.json?download=true"
        taskPath["edinburgh_xsum"] = "https://huggingface.co/tulika214/Quantized_4_bit_models/resolve/main/edinburgh_xsum.json?download=true"
        val task_url = Uri.parse(taskPath[task])
        val mainHandler = Handler(Looper.getMainLooper())
        downloadFile(task_url, task_file) { success, message ->
            mainHandler.post {
                if (success) {
                    messageLog += "\nTask file downloaded, run the benchmark again"
                } else {
                    messageLog += "\nTask file downloading"
                }
            }
        }
    }
    fun downloadFile(uri: Uri, destinationFile: File, callback: (Boolean, String) -> Unit) {
        val client = OkHttpClient()
        val request = Request.Builder().url(uri.toString()).build()

        client.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                callback(false, e.message ?: "Unknown error")
            }

            override fun onResponse(call: Call, response: Response) {
                response.body?.let { body ->
                    try {
                        val inputStream = body.byteStream()
                        val outputStream = FileOutputStream(destinationFile)

                        inputStream.use { input ->
                            outputStream.use { output ->
                                input.copyTo(output)
                            }
                        }

                        callback(true, "File downloaded successfully")
                    } catch (e: Exception) {
                        callback(false, e.message ?: "Error saving file")
                    }
                } ?: callback(false, "Response body is null")
            }
        })
    }
}
