# SpacedRAG
A knowledge corruption attack based off of PoisonedRAG to circumvent the defense measures using Kmeans clustering and Rouge-L scoring proposed by TrustRAG.

## Setup the environment
This project was developed and tested with Python 3.10. Compatibility with other Python versions is not guaranteed.
```
git clone https://github.com/tvogelbacher/SpacedRAG.git
```
```
python -m venv .venv
.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```
```
python -m spacy download en_core_web_sm
```
You will also need PyTorch. To install the correct version for your system, it is best to follow the official instructions at [pytorch.org](https://pytorch.org/get-started/locally/). It might be necessary to uninstall any version of torch and torchvision that was installed as a requirement for another package first.

If you plan on using CUDA (highly recommended) you must of course also have the CUDA toolkit installed on your system. Again, it is best to follow the official instructions at [docs.nvidia.com](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### Download Datasets and NLTK resources
Run the following line to download the datasets worked with in SpacedRAG, as well as some other necessary NLTK resources.
```
python prepare_dataset.py
```

### Set API key
To use a GPT or Gemini model, please set your API key as an environment variable (“OPENAI_API_KEY” or “GEMINI_API_KEY” respectively).
For all Llama models, the key is your HuggingFace Access Token ("HF_ACCESS_TOKEN").
You will also need to log in to Huggingface. To do this, execute the following command in the Python virtual environment and pass your Access Token:
```
hf auth login
```

## Reproduce the results
The repository contains run_*.py scripts in which the relevant hyperparameters can be defined.
```
python run_spaced_gen_adv.py
```
to create adversarial texts following SpacedRAG.
```
python run_vanilla_evaluation.py
```
to evaluate the adversarial texts on a VanillaRAG scheme.
```
python run_trust_evaluation.py
```
to evaluate the adversarial texts on the TrustRAG scheme.
```
python run_evaluate_beir.py
```
to calculate the beir results for your own dataset.

## Acknowledgement
The Code uses:
- [PoisonedRAG](https://github.com/sleeepeer/PoisonedRAG) as the base knowledge corruption attack
- [TrustRAG](https://github.com/HuichiZhou/TrustRAG) as defense against PoisonedRAG
- The [Beir](https://github.com/beir-cellar/beir) Benchmark
- [Contriever](https://github.com/facebookresearch/contriever) for retrieval augmented generation (RAG)
