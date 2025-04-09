# Code-R1: Reproducing R1 for Code with Reliable Rewards

This repository includes implementations to reproduce the R1 pipeline for coding with synthetic datasets.

Some contents in this readme are modified for [KodCode](https://kodcode-ai.github.io/).
## Setup

### Environment

```bash
# For training
pip install -e .
pip install vllm==0.8.2
pip install tensordict==0.6.0
pip install flash-attn --no-build-isolation
pip install wandb IPython matplotlib gpustat # utility
pip install -U "huggingface_hub[cli]"

sudo apt-get install python3-pytest -y # pytest env for kodcode
```

### Sandboxing

Please install `firejail` for reliable sandboxing. 

```bash
sudo add-apt-repository ppa:deki/firejail
sudo apt-get update
sudo apt-get install firejail firejail-profiles

# Alternatively, build from source
cd firejail
sudo apt-get install gawk -y
chmod +x ./configure
chmod +x src/man/mkman.sh
./configure && make && sudo make install-strip
cd ..

```

### Datasets (KodCode-Light)

The current version has 10K RL samples (prompt + tests) at [🤗 KodCode/KodCode-Light-RL-10K](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K):

To produce locally validated RL data:

```bash
python examples/data_preprocess/kodcode.py
```

### Run KodCode!

We have tested several GRPO configurations across various GPU types (including A100, A6000, and RTX 4090).
For example, to fine-tune a 7B model with GRPO using 8×A100 GPUs, simply run:

```bash
bash kodcode_grpo_7b_8a100.sh
```

## Code-R1 based on 7B models + KodCode-Light

**Experimental Setup.** We conduct RL experiments on both Qwen2.5-7B-Instruct-1M and Qwen2.5-Coder-7B-Instruct using 10K randomly selected samples from KODCODE-V1, which we named as [🤗 KodCode/KodCode-Light-RL-10K](https://huggingface.co/datasets/KodCode/KodCode-Light-RL-10K). We perform GRPO with actor learning rate of 5\*10^-7, 16 rollouts per question, a batch size of 256, max response length of 4096, and apply KL coefficient of 0.001.

**Experimental Results.** The experimental results are demonstrated in the table below. We observe significant performance improvement on all benchmarks compared to baselines after RL. In addition, we observe that continuing to increase the training steps can further enhance the performance.

| Model                            | LiveCodeBench (Easy\|Medium\|Hard)          | BCB-Complete (Full\|Hard)   | BCB-Instruct (Full\|Hard)  | HumanEval (Base\|+)| MBPP (Base\|+) | Average |
| -------------------------------- | ----------------------- | -------------- | -------------- | -------------- | -------------- | ------- |
| Qwen2.5-Coder-7B-Instruct | 0.574 \| 0.230 \| 0.044 | 0.520 \| 0.216 | 0.418 \| 0.196 | 0.915 \| 0.854 | 0.831 \| 0.717 | 0.5232  |
| + RL KodCode-10K (Step 128)    | 0.652 \| 0.211 \| 0.041 | 0.525 \| 0.257 | 0.422 \| 0.203 | 0.909 \| 0.860 | 0.849 \| 0.728 | 0.5356  |
| + RL KodCode-10K (Step 256)    | 0.645 \| 0.199 \| 0.033 | 0.537 \| 0.270 | 0.429 \| 0.216 | 0.902 \| 0.854 | 0.865 \| 0.741 | **0.5399**  |
| Qwen2.5-7B-Instruct-1M (Q7I1M)   | 0.577 \| 0.124 \| 0.037 | 0.453 \| 0.142 | 0.366 \| 0.176 | 0.860 \| 0.793 | 0.788 \| 0.693 | 0.4763  |
| + RL KodCode-10K (Step 128)    | 0.602 \| 0.190 \| 0.026 | 0.470 \| 0.196 | 0.367 \| 0.135 | 0.902 \| 0.835 | 0.810 \| 0.709 | 0.4969  |
| + RL KodCode-10K (Step 256)    | 0.570 \| 0.187 \| 0.030 | 0.482 \| 0.196 | 0.368 \| 0.128 | 0.915 \| 0.860 | 0.828 \| 0.728 | **0.503**   |

## Citation

If you find this work helpful...

```bibtex
@article{code-r1,
  title={Code-R1: Reproducing R1 for Code with Reliable Rewards},
  author={Liu, Jiawei and Zhang, Lingming},
  howpublished={\url{https://github.com/ganler/code-r1}},
  year={2025}
}

@article{xu2025kodcode,
      title={KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding}, 
      author={Zhangchen Xu and Yang Liu and Yueqin Yin and Mingyuan Zhou and Radha Poovendran},
      year={2025},
      eprint={2503.02951},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.02951}, 
}
```

## Acknowledgements

* [Verl](https://github.com/volcengine/verl)
* [Logic-RL](https://github.com/Unakar/Logic-RL)

## License

Apache-2.0. See [LICENSE.code-r1](LICENSE.code-r1) for more details.
