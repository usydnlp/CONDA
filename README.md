# [CONDA: a CONtextual Dual-Annotated dataset for in-game toxicity understanding and detection](https://arxiv.org/abs/2106.06213)

## Henry Weld, Guanghao Huang, Jean Lee, Tongshu Zhang, Kunze Wang, Xinghong Guo, Siqu Long, Josiah Poon, Soyeon Caren Han (2021)

## To appear at ACL-IJCNLP 2021

Abstract: Traditional toxicity detection models have focused on the single utterance level without deeper understanding of context. We introduce CONDA, a new dataset for in-game toxic language detection enabling joint intent classification and slot filling analysis, which is the core task of Natural Language Understanding (NLU). The dataset consists of 45K utterances from 12K conversations from the chat logs of 1.9K completed Dota 2 matches. We propose a robust dual semantic-level toxicity framework, which handles utterance and token-level patterns, and rich contextual chatting history. Accompanying the dataset is a thorough in-game toxicity analysis, which provides comprehensive understanding of context at utterance, token, and dual levels. Inspired by NLU, we also apply its metrics to the toxicity detection tasks for assessing toxicity and game-specific aspects. We evaluate strong NLU models on CONDA, providing fine-grained results for different intent classes and slot classes. Furthermore, we examine the coverage of toxicity nature in our dataset by comparing it with other toxicity datasets.

![An example intent/slot annotation from the CONDA (CONtextual Dual-Annotated) dataset.](/resources/figure1_ingame.png "An example intent/slot annotation from the CONDA (CONtextual Dual-Annotated) dataset.")

_For any issue related to the code or data, please first search for solution in Issues section. If your issue is not addressed there, post a comment there and we will help soon._

This repository is for the CONDA dataset as covered in our paper referenced above. 

1. How to get our CONDA dataset?

      --- open the prepare_dataset.ipynb file in the Colab
      
      --- run the code from the top to the auto labeling part
      
      --- get the csv file named 45k_after_slot_annotation.csv
      
2. How to get the inputs for models?

      --- use construct inputs for xxx part in the prepare_dataset.ipynb
      
      --- download seq.in, seq.out, label files and put them into train/valid/test folder

3. How to run the baseline models?

      --- upload the folders in the baseline_models folder to the Google drive
      
      --- open the ipynb files in the Colab
      
      --- set the path to your own path for storing the model folder
      
      --- run the code
      
4. How to get model results?

      --- After running the models, open the experiment_result_analysis.ipynb in the Colab
      
      --- run the corresponding part in the ipynb file
      
5. How to draw graphs?

      --- The graph code is in graph.ipynb and line_graph.ipynb
