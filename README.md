# llm-from-scratch

This repository contains a from-scratch implementation of a simple LLM, a ChatGPT-like model, baseed on rasbt/LLMs-from-scratch. It covers text processing, model construction, training, and a unique extension of the model as part of a final project.

```
gpt2
├── download
├── pretrain
├── generate
│   ├── non-interactive  (pass --prompt)
│   └── interactive      (don't pass --prompt)
├── finetune
│   ├── classifier       (classification finetuning)
│   └── assistant        (instruction finetuning)
├── spam-ham
└── chat-bot
```