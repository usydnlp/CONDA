# [CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding and detection](https://arxiv.org/abs/2106.06213)

## Henry Weld, Guanghao Huang, Jean Lee, Tongshu Zhang, Kunze Wang, Xinghong Guo, Siqu Long, Josiah Poon, Soyeon Caren Han (2021)

## To appear at ACL-IJCNLP 2021

Abstract: Traditional toxicity detection models have focused on the single utterance level without deeper understanding of context. We introduce CONDA, a new dataset for in-game toxic language detection enabling joint intent classification and slot filling analysis, which is the core task of Natural Language Understanding (NLU). The dataset consists of 45K utterances from 12K conversations from the chat logs of 1.9K completed Dota 2 matches. We propose a robust dual semantic-level toxicity framework, which handles utterance and token-level patterns, and rich contextual chatting history. Accompanying the dataset is a thorough in-game toxicity analysis, which provides comprehensive understanding of context at utterance, token, and dual levels. Inspired by NLU, we also apply its metrics to the toxicity detection tasks for assessing toxicity and game-specific aspects. We evaluate strong NLU models on CONDA, providing fine-grained results for different intent classes and slot classes. Furthermore, we examine the coverage of toxicity nature in our dataset by comparing it with other toxicity datasets.

Please enjoy a video presentation covering the main points from our paper:

<p align="center">

[![ACL_video](https://img.youtube.com/vi/qRCPSSUuf18/0.jpg)](https://www.youtube.com/watch?v=qRCPSSUuf18)
      
</p>      

_For any issue related to the code or data, please first search for solution in the Issues section. If your issue is not addressed there, post a comment there and we will help soon._

This repository is for the CONDA dataset as covered in our paper referenced above. 

1. How to get our CONDA dataset?

      --- three .csv files are available in the dataset folder, there are train, validation and test files. Together these make up the ~45k samples described in the paper. 
      
      --- the test data is unlabelled, please see the CodaLab section below for more information.
      
2. HWhat baseline models were used in the paper?

      --- upload the folders in the baseline_models folder to Google drive
      
      --- open the ipynb files in Colab
      
      --- set the path to your own path for storing the model folder
      
      --- run the code
      
3. What other resources are there?

      --- As described in the paper the full lexicons for word level annotation are included in the "resources" directory.
      
![An example intent/slot annotation from the CONDA (CONtextual Dual-Annotated) dataset.](/resources/figure1_ingame.png "An example intent/slot annotation from the CONDA (CONtextual Dual-Annotated) dataset.")

## Codalab

If you are interested in our dataset, you are welcome to join in our Codalab competition leaderboard which will be available in October 2021.

## Citation

```
@inproceedings{weld2021CONDA,
  title={{CONDA}: a {CON}textual {D}ual-{A}nnotated dataset for in-game toxicity understanding and detection},
  author={Henry Weld and Guanghao Huang and Jean Lee and Tongshu Zhang and Kunze Wang and Xinghong Guo and Siqu Long and Josiah Poon and Soyeon Caren Han},
  booktitle={Findings of ACL 2021},
  month = aug,
  year = {2021},
  address = {Bangkok, Thailand},
  publisher = "Association for Computational Linguistics"
}
```
