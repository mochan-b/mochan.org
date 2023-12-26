+++
title = 'Adding Perplexity.ai API support to Langchain'
author = 'Mochan Shrestha'
date = 2023-12-26T15:46:13-05:00
draft = false
+++

We will add support for calling perplexity.ai API from LangChain. This will allow us to use Langchain's numerous features and use the LLMs hosted on perplexity.ai.

The full source code and example is given [here](http://github.com/mochan-b/perplexity-ai-langchain).

You can just copy over the `perplexity_ai_llm.py` file to your langchain project.

## Simple Example

Converting the sample from the API documentation to Langchain version, we have the following:

```python
# Create the LLM
llm = PerplexityAILLM(api_key=PERPLEXITY_API_KEY, model_name="mistral-7b-instruct")
# Call the LLM
response = llm("How many stars are there in our galaxy?")
# Print the response
print(response)
```

## Using with Langchain

To use with langchain instead of OpenAI LLM, just replace 

`llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.0)` 

with 

`llm = PerplexityAILLM(api_key=PERPLEXITY_API_KEY, model_name="mistral-7b-instruct")`

Note that this is just a very simple example. Please report any bugs and issues to the github repo.