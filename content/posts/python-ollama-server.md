+++
title = 'Local LLMs using Ollama Server API with Python'
author = 'Mochan Shrestha'
date = 2023-12-03T23:55:55-05:00
draft = false
+++

We will run local LLM models using python by calling the `Ollama` server API. We will use the `requests` library to make the API calls.

Once `Ollama` is installed, `Ollama` is probably already running. You can check by using `sudo systemctl status ollama` or using the browser to view `http://localhost:11434`. Port 11434 is the default port for `Ollama` server.

Here is a simple example of how to use `Ollama` server API to run a local model.

```python
import requests
import json

# URL for the Ollama server
url = "http://localhost:11434/api/generate"

# Input data (e.g., a text prompt)
data = {
    "model": "mistral",
    "prompt": "What is the capital of Michigan?",
}

# Make a POST request to the server
response = requests.post(url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Process the response
    response_text = response.text

    # Convert each line to json
    response_lines = response_text.splitlines()
    response_json = [json.loads(line) for line in response_lines]
    for line in response_json:
        # Print the response. No line break
        print(line["response"], end="")
else:
    print("Error:", response.status_code)
```

The above code will print the following output:

```
The capital of Michigan is Lansing.
```

![Local OLlama](/images/local-ollama.jpg)

The basic idea is to send a POST request to the server with the input data and the response constains the LLM output. Obviously, the `response` has more data than just the text output but in this example we just output the response text.
