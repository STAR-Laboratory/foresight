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
<br></br>

#### VideoSys

```bash
cd VideoSys
pip install -e .
```



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