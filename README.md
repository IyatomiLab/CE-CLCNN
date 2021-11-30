# End-to-End Text Classification via Image-based Embedding using Character-level Networks

[![CoRR preprint arXiv:1810.03595](http://img.shields.io/badge/cs.CL-arXiv%3A1810.03595-B31B1B.svg)](http://arxiv.org/abs/1810.03595)
[![IEEE Xplore](https://img.shields.io/badge/Accepted-IEEE%20AIPR%20Workshop-%2300629B%09)](https://ieeexplore.ieee.org/document/8707407)

Author: [Shunsuke Kitada](https://scholar.google.co.jp/citations?user=GUzGhQIAAAAJ&hl=ja), Ryunosuke Kotani, Hitoshi Iyatomi

| Proposed CE-CLCNN model [[1](https://arxiv.org/abs/1810.03595)] | Example of data augmentation on image domain with random erasing [[2](https://arxiv.org/abs/1708.04896)] |
|:------:|:------:|
| <img width="300" alt="Screen Shot 2021-11-27 18 11 26" src="https://user-images.githubusercontent.com/11523725/143675360-5a227d7f-fa77-4081-8f03-26695da03324.png"> | ![anim_re](https://user-images.githubusercontent.com/11523725/143675309-0cc7dbbf-d49a-45c3-8c6f-7c8590e1395d.gif) |


Abstract: 
*For analysing and/or understanding languages having no word boundaries based on morphological analysis such as Japanese, Chinese, and Thai, it is desirable to perform appropriate word segmentation before word embeddings. But it is inherently difficult in these languages. In recent years, various language models based on deep learning have made remarkable progress, and some of these methodologies utilizing character-level features have successfully avoided such a difficult problem. However, when a model is fed character-level features of the above languages, it often causes overfitting due to a large number of character types. In this paper, we propose a CE-CLCNN, character-level convolutional neural networks using a character encoder to tackle these problems. The proposed CE-CLCNN is an end-to-end learning model and has an image-based character encoder, i.e. the CE-CLCNN handles each character in the target document as an image. Through various experiments, we found and confirmed that our CE-CLCNN captured closely embedded features for visually and semantically similar characters and achieves state-of-the-art results on several open document classification tasks. In this paper we report the performance of our CE-CLCNN with the Wikipedia title estimation task and analyse the internal behaviour.*

- Preprint: https://arxiv.org/abs/1810.03595
- IEEE Xplore: https://ieeexplore.ieee.org/document/8707407

We recommend that you also check out the following studies related to ours.
- Daif et al. "**[AraDIC: Arabic Document Classification Using Image-Based Character Embeddings and Class-Balanced Loss.](https://aclanthology.org/2020.acl-srw.29/)**" Proceedings of ACL SRW. 2020. Code: https://github.com/mahmouddaif/AraDIC
- Aoki et al. "**[Text Classification through Glyph-aware Disentangled Character Embedding and Semantic Sub-character Augmentation.](https://aclanthology.org/2020.aacl-srw.1/)**" Proceedings of AACL-IJCNLP SRW. 2020. Code: https://github.com/IyatomiLab/GDCE-SSA 

## Install and Run the code

![Python 3.8](https://img.shields.io/badge/python-3.8%2B-brightgreen.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Powered by AllenNLP](https://img.shields.io/badge/Powered%20by-AllenNLP-blue.svg)](https://github.com/allenai/allennlp)

### Install the requirements

```shell
pip install -U pip poetry setuptools
poetry install

# If you want to use CUDA 11+, try the following command:
poetry run poe force-cuda11
```

### Run training the model

- Our `CE-CLCNN` models

```shell
bash bash scripts/run_exps/run_ceclcnn.sh

# or dry-run the scripts
# DRY_RUN=1 bash bash scripts/run_exps/run_ceclcnn.sh
```

- [Liu et al. [ACL'17]](https://arxiv.org/abs/1704.04859) models

```shell
bash scripts/run_exps/run_liu_acl17.sh

# or dry-run the scripts
# DRY_RUN=1 bash scripts/run_exps/run_liu_acl17.sh
```

### Inference with test data using the pre-trained model

- Example of inference using the best model for the Japanese wikipedia title dataset.

```shell
CUDA_VISIBLE_DEVICES=0 allennlp predict \
  output/CE-CLCNN/wiki_title/ja/with_RE_and_WT/model.tar.gz \
  https://github.com/frederick0329/Learning-Character-Level/raw/master/data/ja_test.txt \
  --cuda-device 0 \
  --use-dataset-reader \
  --dataset-reader-choice validation \
  --predictor wiki_title \
  --output-file output/CE-CLCNN/wiki_title/with_RE_and_WT/prediction_result.jsonl \
  --silent
```

If you want to use the following pre-trained model, you should download first and then execute the above command.

```shell
mkdir -p output/CE-CLCNN/wiki_title/ja/with_RE_and_WT
wget https://github.com/IyatomiLab/CE-CLCNN/raw/master/pretrained_models/CE-CLCNN/wiki_title/ja/with_RE_and_WT/model.tar.gz -P output/CE-CLCNN/wiki_title/ja/with_RE_and_WT
```

## Pre-trained models

<table>
    <tr>
        <td><strong><center>Dataset</strong></td>
        <td><strong><center>Language</strong></td>
        <td><strong><center>Model</strong></td>
        <td><strong><center>Pre-trained Model</strong></td>
    </tr>
    <tr>
        <td rowspan="9">Wikipedia Title Dataset [<a href="https://arxiv.org/abs/1704.04859">3</a>]</td>
        <td rowspan="3"><center>Chinese</td>
        <td>CE-CLCNN (<strong>proposed</strong>)</td>
        <td><center>[Download]</td>
    </tr>
    <tr>
        <td><strong>CE-CLCNN w/ RE and WT (proposed)</strong></td>
        <td><center>[Download]</td>
    </tr>
    <tr>
        <td>Visual model [<a href="http://dx.doi.org/10.18653/v1/P17-1188">3</a>]</td>
        <td><center>[Download]</td>
    </tr>
    <tr>
        <td rowspan="3"><center>Japanese</td>
        <td>CE-CLCNN (<strong>proposed</strong>)</td>
        <td><center><a href="https://github.com/IyatomiLab/CE-CLCNN/raw/master/pretrained_models/CE-CLCNN/wiki_title/ja/base/model.tar.gz">[Download]</a></td>
    </tr>
    <tr>
        <td><strong>CE-CLCNN w/ RE and WT (proposed)</strong></td>
        <td><center><a href="https://github.com/IyatomiLab/CE-CLCNN/raw/master/pretrained_models/CE-CLCNN/wiki_title/ja/with_RE_and_WT/model.tar.gz">[Download]</a></td>
    </tr>
    <tr>
        <td>Visual model [<a href="http://dx.doi.org/10.18653/v1/P17-1188">3</a>]</td>
        <td><center><a href="https://github.com/IyatomiLab/CE-CLCNN/raw/master/pretrained_models/liu-acl17/wiki_title/ja/visual/model.tar.gz">[Download]</a></td>
    </tr>
    <tr>
        <td rowspan="3"><center>Korea</td>
        <td>CE-CLCNN (<strong>proposed</strong>)</td>
        <td><center>[Download]</td>
    </tr>
    <tr>
        <td><strong>CE-CLCNN w/ RE and WT (proposed)</strong></td>
        <td><center>[Download]</td>
    </tr>
    <tr>
        <td>Visual model [<a href="http://dx.doi.org/10.18653/v1/P17-1188">3</a>]</td>
        <td><center>[Download]</td>
    </tr>
</table>

## Citation

```bibtex
@inproceedings{kitada2018end,
  title={End-to-end text classification via image-based embedding using character-level networks},
  author={Kitada, Shunsuke and Kotani, Ryunosuke and Iyatomi, Hitoshi},
  booktitle={2018 IEEE Applied Imagery Pattern Recognition Workshop (AIPR)},
  pages={1--4},
  year={2018},
  organization={IEEE},
  doi={10.1109/AIPR.2018.8707407},
}
```

## Reference

- [[1](https://doi.org/10.1109/AIPR.2018.8707407)] S. Kitada, R. Kotani, and H. Iyatomi. "End-to-end Text Classification via Image-based Embedding using Character-level Networks." In Proceedings of IEEE Applied Imagery Pattern Recognition (AIPR) Workshop. IEEE, 2018. doi: https://doi.org/10.1109/AIPR.2018.8707407
- [[2](https://doi.org/10.1609/aaai.v34i07.7000)] Z. Zhong et al. "Random erasing data augmentation." In Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 34. No. 07. 2020. doi: https://doi.org/10.1609/aaai.v34i07.7000
- [[3](http://dx.doi.org/10.18653/v1/P17-1188)] F. Liu et al. "Learning Character-level Compositionality with Visual Features." In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2017. doi: http://dx.doi.org/10.18653/v1/P17-1188
