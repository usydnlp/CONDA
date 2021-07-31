# [CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding and detection](https://arxiv.org/abs/2106.06213)

## Henry Weld, Guanghao Huang, Jean Lee, Tongshu Zhang, Kunze Wang, Xinghong Guo, Siqu Long, Josiah Poon, Soyeon Caren Han (2021)

## To appear at ACL-IJCNLP 2021

Abstract: Traditional toxicity detection models have focused on the single utterance level without deeper understanding of context. We introduce CONDA, a new dataset for in-game toxic language detection enabling joint intent classification and slot filling analysis, which is the core task of Natural Language Understanding (NLU). The dataset consists of 45K utterances from 12K conversations from the chat logs of 1.9K completed Dota 2 matches. We propose a robust dual semantic-level toxicity framework, which handles utterance and token-level patterns, and rich contextual chatting history. Accompanying the dataset is a thorough in-game toxicity analysis, which provides comprehensive understanding of context at utterance, token, and dual levels. Inspired by NLU, we also apply its metrics to the toxicity detection tasks for assessing toxicity and game-specific aspects. We evaluate strong NLU models on CONDA, providing fine-grained results for different intent classes and slot classes. Furthermore, we examine the coverage of toxicity nature in our dataset by comparing it with other toxicity datasets.

![An example intent/slot annotation from the CONDA (CONtextual Dual-Annotated) dataset.](/resources/figure1_ingame.png "An example intent/slot annotation from the CONDA (CONtextual Dual-Annotated) dataset.")

_For any issue related to the code or data, please first search for solution in Issues section. If your issue is not addressed there, post a comment there and we will help soon._

This repository is for the CONDA dataset as covered in our paper referenced above. 

1. How to get our CONDA dataset?

      --- get the csv file named 45k_final.csv from the folder "data". This is the datatset described in the paper.

      --- open the prepare_dataset.ipynb file in the Colab
      
      --- run the code from the top to the auto labeling part
      
      --- get the csv file named 45k_final.csv
      
2. How to get the inputs for models?

      --- use construct inputs for xxx part in the prepare_dataset.ipynb
      
      --- download seq.in, seq.out, label files and put them into train/valid/test folder

3. How to run the baseline models?

      --- upload the folders in the baseline_models folder to Google drive
      
      --- open the ipynb files in Colab
      
      --- set the path to your own path for storing the model folder
      
      --- run the code
      
4. How to get model results?

      --- After running the models, open the experiment_result_analysis.ipynb in Colab
      
      --- run the corresponding part in the ipynb file
      
5. How to do data exploration?

      --- The data exploration code is in graph.ipynb and line_graph.ipynb in the "code" directory

6. What other resources are there?

      --- As described in the paper the full lexicons for word level annotation are included in the "resources" directory.

## Codalab

If you are interested in our dataset, you are welcome to join in the Codalab competition at ...

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

Please enjoy a video presentation covering the main points from our paper:

[![ACL_video](https://img.youtube.com/vi/qRCPSSUuf18/0.jpg)](https://www.youtube.com/watch?v=qRCPSSUuf18)


