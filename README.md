# Zero-shot Cross-lingual Transfer of Prompt-based Tuning with a Unified Multilingual Prompt

Implementation of the paper Zero-shot Cross-lingual Transfer of Prompt-based Tuning with a Unified Multilingual Prompt.

## Acknowledgment

This implementation is based on the released source code of [LM-BFF](https://github.com/princeton-nlp/LM-BFF), and we appreciate their high-quality code implementation, which facilitates subsequent research.

## Requirements

To run our code, please install all the dependency packages by using the following command:

```
pip install -r requirements.txt
```

**NOTE**: Different versions of packages (like `pytorch`, `transformers`, etc.) may lead to different results from the paper. However, the trend should still hold no matter what versions of packages you use.

## Data

We pack the dataset ([Multilingual Amazon Reviews Corpus](https://registry.opendata.aws/amazon-reviews-ml/)) in the `/data/` folder, which contains the link to original dataset (`/data/original/`) and the part sampled by us (`/data/k-shot/`).


**NOTE**: If you need to use their dataset, please cite their [paper](https://arxiv.org/abs/2010.02573).

```bibtex
@inproceedings{keung2020multilingual,
  title={The Multilingual Amazon Reviews Corpus},
  author={Keung, Phillip and Lu, Yichao and Szarvas, Gy{\"o}rgy and Smith, Noah A},
  booktitle={Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  pages={4563--4568},
  year={2020}
}
```


## Run Our Code

We provide scripts for running experiments, see `main_exp.sh`.


## Citation

Please cite our paper if you use UniPrompt in your work:

```bibtex
@article{huang2022zero,
  title={Zero-shot Cross-lingual Transfer of Prompt-based Tuning with a Unified Multilingual Prompt},
  author={Huang, Lianzhe and Ma, Shuming and Zhang, Dongdong and Wei, Furu and Wang, Houfeng},
  journal={arXiv preprint arXiv:2202.11451},
  year={2022}
}
```
