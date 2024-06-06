//
//  ContentView.swift
//  MobileAIBench
//
//  Created by Tulika Awalgaonkar on 6/3/24.
//

import SwiftUI
import SwiftData

struct ContentView: View {
    @StateObject var llamaState = LlamaState()
    @State private var selectedModel: String = "phi-2"
    @State private var selectedTask: String = "hotpot_qa"
    @State private var selectedExample: Int = 10
    
    let models = ["phi-2", "TinyLlama-1.1B-Chat", "gemma-2b-it", "stablelm-zephyr-3b", "llava-phi-2-3b"]
    let text_tasks = ["hotpot_qa", "databricks_dolly", "sql_create_context", "edinburgh_xsum"]
    let MM_tasks = ["vqav2", "scienceqa"]
    let text_examples = [10,50,100]
    let MM_examples = [10,25,50]
    var tasks: [String] {
        switch selectedModel {
        case "phi-2", "TinyLlama-1.1B-Chat", "gemma-2b-it", "stablelm-zephyr-3b":
            return text_tasks
        case "llava-phi-2-3b":
            return MM_tasks
        default:
            return []
        }
    }
    
    var examples: [Int] {
        switch selectedModel {
        case "phi-2", "TinyLlama-1.1B-Chat", "gemma-2b-it", "stablelm-zephyr-3b":
            return text_examples
        case "llava-phi-2-3b":
            return MM_examples
        default:
            return []
        }
    }
        
    var body: some View {
            VStack(spacing: 20) {
                Text("MobileAIBench")
                    .font(.title)
                                .fontWeight(.bold)
                                .padding(.top, 20)
                // Dropdowns in horizontal alignment
                VStack(spacing: 5) {
                    // Models Dropdown
                    VStack {
                        Text("Select Model")
                            .font(.headline)
                        Picker("Select Model", selection: $selectedModel) {
                            ForEach(models, id: \.self) { model in
                                Text(model)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(maxWidth: .infinity)
                        .onChange(of: selectedModel) { newValue in
                                                        updateDefaultTask(for: newValue)
                                                    }
                    }
                    
                    // Task Dropdown
                    VStack {
                        Text("Select Task")
                            .font(.headline)
                        Picker("Select Task", selection: $selectedTask) {
                            ForEach(tasks, id: \.self) { task in
                                Text(task)
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(maxWidth: .infinity)
                        .disabled(tasks.isEmpty)
                    }
                    
                    // Examples Dropdown
                    VStack {
                        Text("Select Example")
                            .font(.headline)
                        Picker("Select Example", selection: $selectedExample) {
                            ForEach(examples, id: \.self) { example in
                                Text("\(example)")
                            }
                        }
                        .pickerStyle(MenuPickerStyle())
                        .frame(maxWidth: .infinity)
                        .disabled(examples.isEmpty)
                    }
                }
                .padding()
                .background(Color(.systemGray6))
                .cornerRadius(10)
                .shadow(radius: 5)
                
                ScrollView(.vertical, showsIndicators: true) {
                    Text(llamaState.messageLog)
                        .font(.system(size: 12))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding()
                        .onTapGesture {
                            UIApplication.shared.sendAction(#selector(UIResponder.resignFirstResponder), to: nil, from: nil, for: nil)
                        }
                }
                // Submit Button
                Button(action: {
                    submitData()
                }) {
                    Text("Run")
                        .padding()
                        .background(Color.blue)
                        .foregroundColor(.white)
                        .cornerRadius(8)
                }
                .disabled(selectedModel.isEmpty || selectedTask.isEmpty || selectedExample == -1)
            }
            .padding()
        }
    
    private func updateDefaultTask(for model: String) {
        switch model {
        case "phi-2", "TinyLlama-1.1B-Chat", "gemma-2b-it", "stablelm-zephyr-3b":
            selectedTask = "hotpot_qa"
        case "llava-phi-2-3b":
            selectedTask = "vqav2"
        default:
            selectedTask = ""
        }
    }
    private func submitData() {
            // Implement the submit action to send data to backend
            let submission = [
                "model": selectedModel,
                "task": selectedTask,
                "example": String(selectedExample)
            ]
            print("Submitted data: \(submission)")
        Task{
            await llamaState.bench_all(name: selectedModel, task_name: selectedTask, examples: selectedExample)
        }
            // Add your backend submission code here
        }
}

#Preview {
    ContentView()
        .modelContainer(for: Item.self, inMemory: true)
    
    
}
