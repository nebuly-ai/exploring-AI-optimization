<p align="center">
  <a href="https://www.nebuly.com/towards-efficient-ai">Subscribe to the newsletter</a> •
  <a href="https://discord.gg/RbeQMu886J">Join the community</a>
</p>

<img height="25" width="100%" src="https://user-images.githubusercontent.com/83510798/211585773-c7610d6f-634c-4ba7-957c-72c3fb5af999.png">


# Weekly insights from top papers on AI

Welcome to our library of the best insights from the best papers on AI and machine learning. A new paper will be added every Friday. <br />
Don't hesitate to [open an issue](https://github.com/nebuly-ai/exploring-AI-optimization/issues) and submit a paper that you found interesting and the 3 key takeaways. 


## Week #3: [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/pdf/2201.11903.pdf)

- The ability of a Large Language Model (LLM) to produce a better quality response is closely related to the prompt used. For example, providing the model with an example or a chain of thoughts (CoT) as a prompt produces a higher quality response without training; this type of technique is usually called InContextLearning, since the model learns from the provided context instead of updating parameters.
- The chain-of-thought prompt is particularly useful in arithmetic reasoning, and it has been shown that for symbolic reasoning, the chain-of-thought prompt facilitates generalization of the out-of-distribution to longer sequences.
- A comparative analysis of the usefulness of chain reasoning versus model size was conducted, showing that larger models are better reasoners and can benefit more from the chain reasoning prompt than smaller models.


## Week #2: [Explanations from Large Language Models Make Small Reasoners Better](https://arxiv.org/pdf/2210.06726.pdf)

- Tuning and inference of LLMs are not trivial in terms of computational cost, so creating smaller models that can be used to solve specific tasks using LLMs as teachers can have several advantages.
- In this case, an LLM is used to produce chain reasoning that is then validated by comparing the final what answer from the LLM with that provided by dataset y. A new dataset {x, e, y} of explanations is then created from a smaller dataset {x, y} containing only question and answer; the new dataset of examples is used to train a smaller T5 3B model to produce the answer along with the chain of thought.
- The results show that the resulting model has comparable performance on the Common Sense Question Answering dataset of GPT3 using Zero-Shot-CoT.


## Week #1: [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/pdf/2211.09110.pdf)

- The Holistic Evaluation of Language Models (HELM) is a toolkit designed to improve the transparency of language models and better understand their capabilities, limitations, and risks.
- HELM uses a multi-metric approach to evaluate language models across a wide range of scenarios and metrics, including accuracy, calibration, robustness, fairness, bias, toxicity, and efficiency.
- HELM conducts a large-scale evaluation of 30 prominent language models across 42 different scenarios, including 21 that have not previously been used in mainstream LM evaluation. The results of the evaluation and all raw model prompts and completions are made publicly available.


<img height="25" width="100%" src="https://user-images.githubusercontent.com/83510798/211585773-c7610d6f-634c-4ba7-957c-72c3fb5af999.png">

<p align="center">
  <a href="https://www.nebuly.com/towards-efficient-ai">Subscribe to the newsletter</a> •
  <a href="https://discord.gg/RbeQMu886J">Join the community</a>
</p>
