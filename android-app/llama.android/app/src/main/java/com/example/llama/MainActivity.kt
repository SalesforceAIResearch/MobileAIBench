package com.example.llama

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.BasicTextField
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalFocusManager
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.TextFieldValue
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import android.os.StrictMode
import android.os.StrictMode.VmPolicy
import androidx.activity.viewModels
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import com.example.llama.ui.theme.LlamaAndroidTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import java.io.File
import okhttp3.*
import java.io.FileOutputStream
import java.io.IOException
import android.net.Uri
import android.os.Handler
import android.os.Looper

class MainActivity(
): ComponentActivity() {
    private val tag: String? = this::class.simpleName

    private val viewModel: MainViewModel by viewModels()



    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        StrictMode.setVmPolicy(
            VmPolicy.Builder(StrictMode.getVmPolicy())
                .detectLeakedClosableObjects()
                .build()
        )


        val extFilesDir = getExternalFilesDir(null)
        println("flora")
        println(extFilesDir)


        setContent {
            LlamaAndroidTheme {
                // A surface container using the 'background' color from the theme
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    if (extFilesDir != null) {
                        MainCompose(
                            viewModel,
                            extFilesDir
                        )
                    }
                }

            }
        }
    }
}

@Composable
fun MainCompose(
    viewModel: MainViewModel,
    extFilesDir: File
) {
    var selectedModel by remember { mutableStateOf("phi-2") }
    var selectedTask by remember { mutableStateOf("hotpot_qa") }
    var selectedExample by remember { mutableStateOf(10) }

    val models = listOf("phi-2", "TinyLlama-1.1B-Chat", "gemma-2b-it")
    val textTasks = listOf("hotpot_qa", "databricks_dolly", "sql_create_context", "edinburgh_xsum")
    val textExamples = listOf(10, 50, 100)

    val tasks = when (selectedModel) {
        "phi-2", "TinyLlama-1.1B-Chat", "gemma-2b-it" -> textTasks
        else -> emptyList()
    }

    val examples = when (selectedModel) {
        "phi-2", "TinyLlama-1.1B-Chat", "gemma-2b-it" -> textExamples
        else -> emptyList()
    }

    val focusManager = LocalFocusManager.current

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            text = "MobileAIBench",
            fontSize = 24.sp,
            fontWeight = FontWeight.Bold,
            modifier = Modifier
                .padding(top = 20.dp)
                .fillMaxWidth()
                .wrapContentWidth(Alignment.CenterHorizontally)
        )

        Spacer(modifier = Modifier.height(20.dp))

        Column(modifier = Modifier.padding(16.dp).background(Color.Gray.copy(alpha = 0.1f)).padding(16.dp)) {
            // Model Dropdown
            DropdownMenuWithLabel(
                label = "Select Model",
                items = models,
                selectedItem = selectedModel,
                onItemSelected = {
                    selectedModel = it
                    updateDefaultTask(it, onTaskUpdate = { selectedTask = it })
                }
            )

            Spacer(modifier = Modifier.height(5.dp))

            // Task Dropdown
            DropdownMenuWithLabel(
                label = "Select Task",
                items = tasks,
                selectedItem = selectedTask,
                onItemSelected = { selectedTask = it },
                enabled = tasks.isNotEmpty()
            )

            Spacer(modifier = Modifier.height(5.dp))

            // Example Dropdown
            DropdownMenuWithLabel(
                label = "Select Example",
                items = examples.map { it.toString() },
                selectedItem = selectedExample.toString(),
                onItemSelected = { selectedExample = it.toInt() },
                enabled = examples.isNotEmpty()
            )
        }

        Spacer(modifier = Modifier.height(10.dp))

        val scrollState = rememberScrollState()
        BasicTextField(
            value = TextFieldValue(viewModel.messageLog),
            onValueChange = {},
            modifier = Modifier
                .fillMaxWidth()
                .height(300.dp)
                .verticalScroll(scrollState)
                .background(Color.Gray.copy(alpha = 0.1f))
                .padding(8.dp)
        )

        Spacer(modifier = Modifier.height(20.dp))

        Button(
            onClick = { viewModel.submit_data(selectedModel, selectedTask, selectedExample, extFilesDir) },
            enabled = selectedModel.isNotEmpty() && selectedTask.isNotEmpty() && selectedExample != -1,
            modifier = Modifier.align(Alignment.CenterHorizontally)
        ) {
            Text(text = "Run")
        }
    }
}

@Composable
fun DropdownMenuWithLabel(
    label: String,
    items: List<String>,
    selectedItem: String,
    onItemSelected: (String) -> Unit,
    enabled: Boolean = true
) {
    Column {
        Text(text = label, fontSize = 18.sp, fontWeight = FontWeight.Bold)
        var expanded by remember { mutableStateOf(false) }
        Box {
            Text(
                text = selectedItem,
                modifier = Modifier
                    .fillMaxWidth()
                    .background(Color.White)
                    .padding(12.dp)
                    .clickable(enabled = enabled) { expanded = true }
            )
            DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
                items.forEach { item ->
                    DropdownMenuItem(text = { Text(text = item) }, onClick = {
                        expanded = false
                        onItemSelected(item)
                    })
                }
            }
        }
    }
}

fun updateDefaultTask(model: String, onTaskUpdate: (String) -> Unit) {
    val defaultTask = when (model) {
        "phi-2", "TinyLlama-1.1B-Chat", "gemma-2b-it" -> "hotpot_qa"
        else -> ""
    }
    onTaskUpdate(defaultTask)
}

