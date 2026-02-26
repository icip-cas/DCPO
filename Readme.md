# 🚀 DCPO

This repository contains the official implementation of the paper:  
**Decoupling Reasoning and Confidence:Resurrecting Calibration in Reinforcement Learning from Verifiable Rewards**

## Environment Setup

The code has been successfully tested on **8 × 80GB A100 GPUs** with **CUDA 12.8**.  
To create a Conda environment, run the following commands:

```bash
git clone https://github.com/mazhengzhao/DCPO.git
cd DCPO
conda env create -f environment.yml
```
---

## Running the Code

After setting up the environment, run the following command to start training:

```bash
bash examples/Qwen3-8B.sh
```

---

## Evaluating Metrics

To compute evaluation metrics such as **Accuracy**, **Expected Calibration Error (ECE)**, **Brier Score (BS)** and **Positive Calibration Error (PCE)**, deploy a vllm service of your model, identify your model name and service url and run:

```bash
bash examples/eval.sh
```

The script will log output in folder logs/$model_name/ and plot a calibration curve in Figs/{model_name}.

---
## 🙏 Acknowledgements

This repository builds upon the following open-source projects, to which we are deeply grateful:
[verl](https://github.com/volcengine/verl), [AR-Lopti](https://github.com/zhyang2226/AR-Lopti), [LogicRL](https://github.com/Unakar/Logic-RL), [DeepScaleR](https://github.com/agentica-project/rllm), [AdaRFT](https://github.com/limenlp/verl), [CCGSPG](https://github.com/HaotianLiu123/CCGSPG)
