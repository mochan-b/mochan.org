+++
title = 'Local LLMs using Llama.Cpp and Python'
author = 'Mochan Shrestha'
date = 2023-12-19T10:36:20-05:00
draft = false
+++

We will run local GMML models using llama.cpp and python. We will use the python bindings for the llama.cpp library called `llama-cpp-python` which can be installed using `pip`.

We are responsible for finding and downloading the desired GMML models. There is no built in mechanism to download them (like ollama server). The files that work for the ollama server do work on llama.cpp as well and that can be used.

![Llama CPP](/images/llama-cpp.png)

## Simple Example

Here is a simple example of how to use llama.cpp to run a local model.

```python
from llama_cpp import Llama
llm = Llama(model_path="<path to model>")
output = llm(
      "Q: What is standard deviation in statistics.' A: ", # Prompt
      echo=True, # Echo the prompt back in the output
)
```

## Constraining to JSON Output

`llama.cpp` also has GBNF (GGML BNF (Backus-Naur Form)) format for the output which constrains the output to be of a certain format. To use this to force JSON format, we load the JSON grammar which can be found [here](https://github.com/ggerganov/llama.cpp/blob/master/grammars/json.gbnf). 

The code to use the JSON grammar is as follows:

```python
grammar = LlamaGrammar.from_file('json.gbnf')
output = llm(
      "Q: What are the key concepts to learn in statistics?. Please give a JSON list of the concepts. A: ", # Prompt
      echo=True, # Echo the prompt back in the output
      grammar=grammar, # Grammar file to use
)
```

## Running on the GPU

If you install `llama-cpp-python` using `pip`, it by default will install the CPU version of the library. 

In case the CPU version is installed, you have to remove the currently installed version and install the GPU version. 

```bash
pip uninstall -y llama-cpp-python
CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python
```

It might fail if you do not have the nVidia CUDA toolkit installed in the path. For me using ubuntu, the installation can be done following the instructions [here](https://developer.nvidia.com/cuda-12-2-0-download-archive) for CUDA 12.2. Please get the right version for your system accordingly. `nvcc --version` should show the version of CUDA installed or otherwise add it to the path using 

```bash
export PATH=/usr/local/cuda/bin:$PATH 
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

The model can now be loaded by specifying the `n_gpu_layers`paramter. The rest of the code remains the same.

A good way to check if the GPU is being used is to use `nvidia-smi` command. It should show memory being used to load the LLM.

```python
llm = Llama(model_path="<path to model>", n_gpu_layers=40)
```