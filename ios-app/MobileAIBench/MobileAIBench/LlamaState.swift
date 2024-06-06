//
//  LlamaState.swift
//  LLMBench
//
//  Created by Tulika Awalgaonkar on 4/10/24.
//

import Foundation
import llamacpp_framework

struct Question: Codable {
    let imageID: String
    let prompt: String
    
    enum CodingKeys: String, CodingKey {
        case imageID = "image_id"
        case prompt
    }
}

typealias Questions = [String: Question]

class LlamaState: ObservableObject {

    @Published var messageLog = ""
    var llm = MultiModal()
    private var llamaContext: LlamaContext?
    let NS_PER_S = 1_000_000_000.0
    init() {
        print("IN START")
    }
    
    func multi_inference(model: String, projector: String, model_name: String, no_of_examples:Int, task_name:String){
        if let imagesDirectoryUrl = Bundle.main.url(forResource: task_name+"/images", withExtension: nil) {
            print("Images directory found: \(imagesDirectoryUrl)")
        } else {
            print("Images directory not found")
            messageLog+="\nERROR: Images directory not found"
            return
        }
        let task_image_path=task_name+"/images"
        print(task_image_path)
        if let fileURL = Bundle.main.url(forResource: "prompts", withExtension: "json", subdirectory: task_name){
            do{
                let jsonData = try Data(contentsOf: fileURL)
                let questions = try JSONDecoder().decode(Questions.self, from: jsonData)
                var count=0
                    // Access the parsed data
                let t_start = DispatchTime.now().uptimeNanoseconds
                llm.loadModel(model)
                let t_heat_end = DispatchTime.now().uptimeNanoseconds
                messageLog+="\nLoaded model \(model_name)"
                let t_heat = Double(t_heat_end - t_start) / NS_PER_S
                var final_completion=""
                
                messageLog+="\nStarting evaluation...\n"
                for (key, question) in questions {
                    if count>=no_of_examples{
                        break
                    }
                    messageLog+="\nQuestion: \(question.prompt)"
                    let imagePath = Bundle.main.url(forResource: key, withExtension: "jpg", subdirectory: task_image_path)?.path
                    print(imagePath)
                    var completion=""
                    llm.evaluateMultimodal(question.prompt, usingClipModelAtPath: projector, modelAtPath: model, imageAtPaths: [imagePath]){ (value, isComplete, isError) in
                        if isComplete{
                            //print(value)
                            completion=value ?? ""
                        }
                    }
                    count+=1
                    messageLog+="\nAnswer: "+completion
                    print("Image ID: \(question.imageID)")
                    print("Prompt: \(question.prompt)")
                    final_completion+=completion
                }
                let t_end = DispatchTime.now().uptimeNanoseconds
                let t_generation = Double(t_end - t_heat_end) / NS_PER_S
                let words = final_completion.split { $0.isWhitespace}
                let numberOfWords = words.count
                print(final_completion)
                print(numberOfWords)
                let tokens_per_second = Double(numberOfWords) / t_generation
                messageLog += """
                    \n
                    Done
                    Model load time \(t_heat) sec
                    Model generation time \(t_generation)
                    Tokens generated: \(numberOfWords)
                    Generated \(tokens_per_second) token/sec\n
                    """
            }catch {
                print("Error reading JSON file: \(error)")
                messageLog+="\nERROR: Error reading JSON file"
            }

        }else {
            print("JSON file not found.")
            messageLog+="\nERROR: JSON file not found."
        }
        
        print("Finished")
        
    }
    func loadModel(modelUrl: URL?) throws {
        print("load model")
        if let modelUrl {
            print(modelUrl)
            messageLog += "\nLoading model...\n"
            llamaContext = try LlamaContext.create_context(path: modelUrl.path())
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"

        } else {
            messageLog += "Load a model from the list below\n"
        }
    }
    
    func eval_model(model: String, dataset:String, model_name: String, no_of_examples:Int, include_context:Bool) async{
        guard let llamaContext else {
            return
        }
        let myDict: [String: String] = ["tinyllama-1.1b-chat_Q4_K_M.gguf": "<|system|>\n{system}</s>\n<|user|>\n{prompt}</s>\n<|assistant|>", "phi-2_Q4_K_M.gguf": "{system}\nInstruct:{prompt}\nOutput:", "gemma-2b-it_Q4_K_M.gguf":"<start_of_turn>user\n{system}\n{prompt}<end_of_turn>\n<start_of_turn>model\n", "stablelm-zephyr-3b_Q4_K_M.gguf": "<|user|>\n{system}\n{prompt}<|endoftext|>\n<|assistant|>\n"]
        var SYS = get_SYS_prompt(dataset: dataset, include_context: include_context)
        var context=""
        if include_context==true{
            context="with context"
        }
        else{
            context="without context"
        }
        if let fileURL = Bundle.main.url(forResource: dataset, withExtension: "json", subdirectory: "datasets") {
            do {
                // Read the JSON data from the file
                let jsonData = try Data(contentsOf: fileURL)
                
                // Parse the JSON data
                if let jsonArray = try JSONSerialization.jsonObject(with: jsonData, options: []) as? [[String: Any]] {
                    // Iterate over each dictionary in the JSON array
                    var count = 0
                    var sumMetric1 = 0.0
                    var sumMetric2 = 0.0
                    for jsonDict in jsonArray {
                        // Extract question and answer from each dictionary
                        if count>=no_of_examples{
                            break
                        }
                        if var question = jsonDict["question"] as? String, let actual_answer = jsonDict["answer"] as? String {
                            // Do something with the question and answer
                            if let dictValue = myDict[model_name] {
                                var new_sys=""
                                var new_ques=""
                                (new_sys, new_ques) = get_prompt(dataset: dataset, question: question, SYS: SYS, include_context: include_context, con: jsonDict["context"] as! String)
                                let prompt1 = dictValue.replacingOccurrences(of: "{system}", with: new_sys)
                                let prompt = prompt1.replacingOccurrences(of: "{prompt}", with: new_ques)
                                //print(prompt)
                                await llamaContext.completion_init(text: prompt)
                                var expected_answer=""
                                var metric1=0.0
                                var metric2=0.0
                                var result=""
                                var isdone=false
                                while await llamaContext.n_cur < (llamaContext.n_len + llamaContext.n_start) {
                                    (result, isdone) = await llamaContext.completion_loop()
                                    if isdone==true{
                                        expected_answer += "\(result)"
                                        break
                                    }
                                    expected_answer += "\(result)"
                                }
                                await llamaContext.clear()
                                print(expected_answer)
                                let llama_timings = await llamaContext.get_llama_timings()
                                print(llama_timings)
                                (metric1,metric2)=task_specific_metric(dataset: dataset, actual: actual_answer, predicted: expected_answer)
                                sumMetric1+=metric1
                                sumMetric2+=metric2
                                
                            } else {
                                print("Invalid model")
                                messageLog+="\nERROR: Model not found, make sure that models in local directory are named as follows: \ntinyllama-1.1b.gguf\nphi-2.Q4_K_M.gguf\ngemma-2b-it.Q4_K_M.gguf"
                            }
                        }
                        count += 1
                        print(count)
                    }
                    //print avg here
                    if count > 0 {
                        print(count)
                        let llama_timings = await llamaContext.get_llama_timings()
                        print(llama_timings)
                        let total_time = llama_timings.t_end_ms-llama_timings.t_start_ms
                        let PromptTPS = 1e3 / llama_timings.t_p_eval_ms * Double(llama_timings.n_p_eval)
                        let EvalTPS = 1e3 / llama_timings.t_eval_ms * Double(llama_timings.n_eval)
                        let SampleTPS = 1e3 / llama_timings.t_sample_ms * Double(llama_timings.n_sample)
                        
                        let model_load_time = llama_timings.t_load_ms / 1000.0
                        let averageTotalTime = total_time / (Double(count)*1000.0)
                        let averageSampleTime = Double(llama_timings.t_sample_ms) / (Double(count)*1000.0)
                        let averagePromptTime = Double(llama_timings.t_p_eval_ms) / (Double(count)*1000.0)
                        let averagePromptTokens = Double(llama_timings.n_p_eval)/Double(count)
                        let averagePromptTokenPerSec=PromptTPS
                        let averageEvalTime=Double(llama_timings.t_eval_ms) / (Double(count)*1000.0)
                        let averageEvalTokens=Double(llama_timings.n_eval)/Double(count)
                        let averageEvalTokenPerSec=EvalTPS
                        let averageMetric1 = sumMetric1/Double(count)
                        let averageMetric2 = sumMetric2/Double(count)
                        let fstring = """
                            \nModel load time: \(model_load_time) sec
                            \nAverage values on \(model_name) for \(dataset) dataset(\(context)) \(count) examples:
                            Number of input tokens: \(averagePromptTokens)
                            Time to first token \(averagePromptTime) sec
                            Input tokens per sec: \(averagePromptTokenPerSec)
                            Sample time \(averageSampleTime) sec
                            Sample tokens per sec: \(SampleTPS)
                            Number of output tokens \(averageEvalTokens)
                            Output eval time \(averageEvalTime) sec
                            Output token per sec: \(averageEvalTokenPerSec)
                            Total time \(averageTotalTime) sec
                            """
                        messageLog += fstring
                        messageLog += print_task_specific_metric(dataset: dataset, metric1: averageMetric1, metric2: averageMetric2)
                    } else {
                        print("No valid data found.")
                        messageLog+="\nERROR: No valid data found."
                    }
                } else {
                    print("JSON data is not in the expected format.")
                    messageLog+="\nERROR: JSON data is not in the expected format."
                }
            } catch {
                print("Error reading JSON file: \(error)")
                messageLog+="\nERROR: Error reading JSON file"
            }
        } else {
            print("JSON file not found.")
            messageLog+="\nERROR: JSON file not found."
        }
    }
    func bench_all(name: String, task_name:String, examples:Int) async{
        messageLog += "\t  ***RUNNING BENCHMARKING***\n"
        let modelDictionary  = ["phi-2": "phi-2_Q4_K_M", "TinyLlama-1.1B-Chat":"tinyllama-1.1b-chat_Q4_K_M", "gemma-2b-it":"gemma-2b-it_Q4_K_M", "stablelm-zephyr-3b": "stablelm-zephyr-3b_Q4_K_M", "llava-phi-2-3b":"llava-phi-2"]
        let model_name = modelDictionary[name]
        let documentsURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let model_path = documentsURL.appendingPathComponent(model_name!+".gguf")
        print(model_path)
        let fileManager = FileManager.default
        if fileManager.fileExists(atPath: model_path.path) {
            print("File exists")
            let model_url_name = model_path.lastPathComponent
            print(model_url_name)
            if name=="llava-phi-2-3b"{
                messageLog+="\nRunning Multimodal eval for \(task_name)"
                let projector_name = documentsURL.appendingPathComponent(model_name!+"-mmproj.gguf")
                print(projector_name)
                if fileManager.fileExists(atPath: projector_name.path){
                    multi_inference(model: model_path.path(), projector: projector_name.path(),model_name: model_url_name, no_of_examples: examples, task_name: task_name)
                }else{
                    messageLog+="\nERROR: Projector file not found, make sure that projector in local directory is named as \(projector_name)"
                }
                
            }else{
                do{
                    try loadModel(modelUrl: model_path)
                    await eval_model(model: model_path.path(), dataset: task_name, model_name:model_url_name, no_of_examples: examples, include_context: true)
                }catch{
                    print("error")
                    messageLog+="\nEncountered unexpected ERROR"
                }
            }
        } else {
            print("File does not exist")
            messageLog+="\nERROR: Model not found, make sure that model in local directory is named as \(model_name)"
        }
    }
}
