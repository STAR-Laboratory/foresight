# Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation

**Foresight** proposes adaptive coarse grained reuse framework for accelerating text-to-video generation models while maintaining video quality.

This repository contains the source code implementation of [Foresight](https://foresight_arXiv.co/).

This source code is available under the [Apache 2.0 License](LICENSE).

## ⚙️ Environment Setup

#### conda Environment
You can create a new conda environment using script.
```bash
conda env create -n foresight-env python=3.10 -y
conda activate foresight-env
```

```bash
pip install -e .
```

## 💻 System Requirements

Right now, **Foresight** has been tested on a 1xA100 node for `Open-Sora`, `Latte` and
`CogVideoX` models on single GPU.

We welcome contributions to evaluate **Foresight** across different models.

## 🏁 Using Foresight

### Supported Models

Currently Foresight supports Open-Sora, Latte and CogVideoX models.

### Foresight Configuration

Foresight requires configuring below parameters to control the ***warmup phase*** and ***reuse phase*** reuse.

#### Parameters

- **warmup**: No of denoising steps used during warmup phase.
  - Type: Integer

- **recalculate**: Mandatory computation interval.
  - Format: Integer

- **threshold**: Scaling factor for threshold.
  - Type: Float

#### Example Configuration

```yaml
warmup: 5
recalculate: 2
threshold: 0.5
```
#### Example Runs

```
cd examples/open_sora
python sample.py
```

```
cd examples/latte
python sample.py
```

```
cd examples/cogvideox
python sample.py
```


## Thank You

Foresight has been implemented on top of [VideoSys](https://github.com/NUS-HPC-AI-Lab/VideoSys), an easy and efficient system for video generation.
<br></br>

## 📝 Citation
```
@article{foresight,
  title={Foresight: Adaptive Layer Reuse for Accelerated and High-Quality Text-to-Video Generation},
  author={Adnan, Muhammad and Kurella, Nithesh and Arunkumar, Akhil and Nair, Prashant},
  year={2025},
  eprint={XXXX.XXXXX},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/XXXX.XXXXX},
}
```