## serves as the reading notes while browsing through the book.

---
12-30-2024 start reading

---
# Chapter 1
1-2-2024

1.1
What is LLM (large language model) ?  An LLM is a neural network designed to understand, generate, and respond to human-like text. LLMs utilize an architecture called the *transformer*, which allows them to pay selective attention to different parts of the input when making predictions, making them especially adept at handling nuances and complexities of human languages.

Machine learning involves development of algorithms that can learn from and make predictiosn or decisions based on data without being explicitly programmed. In contrast to traditional machine learning, deep learning does not identify and select the most relevant features for a deep learning model. 

1.2
LLMs are invaluable for automating task involving parsing and genearting text. 

1.3
Coding an LLM from ground up is an excellent exercise to understand its mechanics and limitations. Custom built LLMs can outperform general-purpose LLMs. Smaller, custom LLMs enable deployment directly on customer devices. Local implementation can significantly decrease latency and reduce server-related costs. 

General process of creating an LLM includes pretraining and fine-tuning. Pre in pretraining refers to the initial phase where a model like an LLM is trained on a large, diver dataset to develop a broad understanding of language. This pretrained model can be further refined through fine-tining, a process where the model is specifically trained on a narrower dataset to particular tasks or domains. 

First step to create an LLM is to train it on raw text, regular text without any labeling information. First training stage of an LLM is known as *pretraining*, creating an initial pretrained LLM, called a base or fundation model. We can further train LLM on labeled data, known as fine-tuning. Two most popular categories of fine-tuning LLMs are *instruction fine-tuning*, labeled dataset consists of instruction and answer pairs, and *classification fine-tuning*, labeled dataset consists of text and associated class labels. 

1.4
Most modern LLMs rely on transformer architecture - [Attention is All You Need](https://arxiv.org/pdf/1706.03762). Transformer architecture consists of two submodules: an ecoder and a decoder. Encoder module processes the input text and encodes it into a series of numerical representations or vectors that capture the contextual information of the input. Decoder module takes these encoded vectors and generates the output text. Both encoder and decoder consist of many layers connected by a so-called self-attention mechanism, which allows the model to weigh importance of different words or tokens in a sequence relative to each other. 

BERT (bidirectional encoder representations from transformers) is built upon original transformer's encoder submodule, specialize in masked word prediction.

GPT (generative pretrained transformers) focuses on decoder portion of transformer architecture and is designed for tasks that require generating texts, adept at zero-shot, ability to generalize to completely unseen tasks without any prior examples and few-shot learning tasks, minimal number of examples user provdes as input. 

1.5
Token is a unit of text that a model reads and the number of tokens in a dataset is roughly equivalent to the number of words and punctuations characters in the text. 

After implementing pretraining code, we will learn how to reuse openly available model weights and load the minto architecture we will implement. 

1.6
GPT was introduced in the paper [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf). GPT-3 is a sclaed-up version of this model that has more parameters and was trained on a larger dataset. 

General GPT architecture is essentially just the decoder part. Decoder style models like GPT generate text by predicting text one word at a time, they are considered a type of *autoregressive* model, incorporating their previous outputs as inputs for future predictions. 

1.7
Building LLM in 3 stages
- stage 1, code attention mechanism 
- stage 2, code and pretraining GPT like LLM
- stage 3, take pretrained LLM and fine-tune it to follow instructions

---
# Chapter 2

2.1
Deep neural network models cannot process raw text directly, as text is categorical, it's not comtapible tiwht mathematical operations used to implement and train neural networks. We need a way to represent words as continuous-valued vectors.

Converting data into a vector format is referred to as *embedding*. Embedding is a mapping from discrete objects to a continuous vector space. Sentence or paragraph embeddings are popular choices for retrieval-augmented generation, combining generation with retrieval to pull relevant information when generating text. 

One popular example is Word2Vec approach, by predicting context of a word given target word or vice versa. 


2.2
Tokens are either individual words or special characters, including punctuation characters. 
Example text - [The Verdict](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/the-verdict.txt)

```
with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()
print("Total number of character:", len(raw_text))
print(raw_text[:99])

import re
text = "Hello, world. Is this-- a test?"
result = re.split(r'([,.:;?_!"()\']|--|\s)', text)
result = [item.strip() for item in result if item.strip()]
print(result)


with open("the-verdict.txt", "r", encoding="utf-8") as f:
	raw_text = f.read()
	processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
	processed = [item.strip() for item in processed if item.strip()]
print(len(processed))
print(processed[:30])
```

2.3
Convert tokens from string to an integer representation to produce token IDs. To convert outputs of an LLM from numbers back into text, we need a way to turn token IDs into text. 

Implement a simple text tokenizer

related code - https://github.com/rasbt/LLMs-from-scratch/blob/main/ch02/01_main-chapter-code/ch02.ipynb

2.4
Modify tokenizer to handle unknown words, and special tokens handling context or other relevant informatin in text. Special tokens include markers for unknown words and document boundaries. Example: <|unk|> and <|endoftext|>

Some additional special tokens:
- [BOS] beginning of sequence
- [EOS] end of sequence
- [PAD] padding

2.5 
A more sophisiticated tokenization scheme - Byte pair encoding (BPE). Public library available - *tiktoken*.























