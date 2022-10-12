# [CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding and detection](https://arxiv.org/abs/2106.06213)

## Henry Weld, Guanghao Huang, Jean Lee, Tongshu Zhang, Kunze Wang, Xinghong Guo, Siqu Long, Josiah Poon, Soyeon Caren Han (2021)
## University of Sydney, NLP Group

## To appear at ACL-IJCNLP 2021

Abstract: Traditional toxicity detection models have focused on the single utterance level without deeper understanding of context. We introduce CONDA, a new dataset for in-game toxic language detection enabling joint intent classification and slot filling analysis, which is the core task of Natural Language Understanding (NLU). The dataset consists of 45K utterances from 12K conversations from the chat logs of 1.9K completed Dota 2 matches. We propose a robust dual semantic-level toxicity framework, which handles utterance and token-level patterns, and rich contextual chatting history. Accompanying the dataset is a thorough in-game toxicity analysis, which provides comprehensive understanding of context at utterance, token, and dual levels. Inspired by NLU, we also apply its metrics to the toxicity detection tasks for assessing toxicity and game-specific aspects. We evaluate strong NLU models on CONDA, providing fine-grained results for different intent classes and slot classes. Furthermore, we examine the coverage of toxicity nature in our dataset by comparing it with other toxicity datasets.

Please enjoy a video presentation covering the main points from our paper:

<p align="centre">

[![ACL_video](https://img.youtube.com/vi/qRCPSSUuf18/0.jpg)](https://www.youtube.com/watch?v=qRCPSSUuf18)
      
</p>      

_For any issue related to the code or data, please first search for solution in the Issues section. If your issue is not addressed there, post a comment there and we will help soon._

This repository is for the CONDA dataset as covered in our paper referenced above. 

1. How to get our CONDA dataset?

      --- three .csv files are available in the dataset folder, there are train, validation and test files. Together these make up the ~45k samples described in the paper. 
      
      --- the test data is unannotated, please see the CodaLab section below for more information.
      
2. What baseline models were used in the paper?

      --- Joint BERT, (Castellucci et al., 2019): https://github.com/monologg/JointBERT
      
      --- Capsule NN, (Zhang et al., 2019): https://github.com/czhang99/Capsule-NLU
      
      --- RNN-NLU, (Liu + Lane, 2016): https://github.com/HadoopIt/rnn-nlu
      
      --- Slot-gated, (Goo et al., 2018) https://github.com/MiuLab/SlotGated-SLU
      
      --- Inter-BiLSTM (Wang et al., 2018): https://github.com/ray075hl/Bi-Model-Intent-And-Slot
      
3. What other resources are there?

      --- As described in the paper the full lexicons for word level annotation are included in the "resources" directory.

<p align="center">
  <img width="600" src="/resources/figure1_ingame.png">
</p>

## Codalab

If you are interested in our dataset, you are welcome to join in our Codalab competition leaderboard which will be available in October 2021.

### Evaluation Metrics
**JSA**(Joint Semantic Accuracy) is used for ranking. An utterance is deemed correctly analysed only if both utterance-level and all the token-level labels including Os are correctly predicted.

Besides, the f1 score of **utterance-level** E(xplicit) and I(mplicit) classes, **token-level** T(oxicity), D(ota-specific), S(game Slang) classes will be shown on the leaderboard (but not used as the ranking metric).

## Citation

```
@inproceedings{weld-etal-2021-conda,
    title = "{CONDA}: a {CON}textual Dual-Annotated dataset for in-game toxicity understanding and detection",
    author = "Weld, Henry  and
      Huang, Guanghao  and
      Lee, Jean  and
      Zhang, Tongshu  and
      Wang, Kunze  and
      Guo, Xinghong  and
      Long, Siqu  and
      Poon, Josiah  and
      Han, Caren",
    booktitle = "Findings of the Association for Computational Linguistics: ACL-IJCNLP 2021",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.findings-acl.213",
    doi = "10.18653/v1/2021.findings-acl.213",
    pages = "2406--2416",
}
```
