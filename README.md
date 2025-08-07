# CL-TASD-MvP

This is the source code for various unilingual-TASD, XL-TASD and ML-TASD models that are described in the paper "Using Contrastive Learning, Multi-view Prompting, and LLMs in Multilingual Term-Aspect Sentiment Detection".

## Data

First, create a `data/sem_eval_raw` directory and download the [SemEval-2016](http://alt.qcri.org/semeval2016/task5/index.php?id=data-and-tools) dataset for the languages used for testing and training. Note here that we use the Subtask 1 data for respective languages. Then rename the SemEval datasets as follows:

- `data/sem_eval_raw`
    - `SemEval16_Restaurants_Test_Language.xml`
    - `SemEval16_Restaurants_Train_Language.xml`

Also create a `data/txt_raw`. Download the restaurant data from the M-ABSA dataset, found in the [M-ABSA GitHub](https://github.com/swaggy66/eval-llm-mabsa.git). Please rename these as follows:

- `data/txt_raw`
    - `SemEval16_Restaurants_Dev_Language.txt`
    - `SemEval16_Restaurants_Test_Language.txt`
    - `SemEval16_Restaurants_Train_Language.txt`

Note that these directory addresses may have to be adapted in the code itself, as the directories are currently set for a combined usage of Google Colab and Google Drive. The current form in the Colab optimised code is as follows: `/content/drive/MyDrive/data/sem_eval_raw`

## Running the GitHub

This entire GitHub can be run with [Google Colab](https://colab.research.google.com/). Note that if you choose to not run these models with Colab, you will have to modify various addresses, of functions, such as the input and output paths of various directories. Also note that the CL-TASD-MvP model in its current form requires an A100 GPU with 40GB of VRAM to run. We cannot guarantee that GPUs with less RAM have the capacity to train the models.

Run our code in the following order, using the `.ipynb` files that are found in `all_shells`.

### Data

For all our data to be formatted correctly, please use the `0_Data_CL_MvP_TASD.ipynb` file. Please run "Notebook Setup first. 

Run the section called "SemEval-2016 Conversion", which converts all the `xml` files into standard `txt` files. Then run the section called "Extract Information from .txt Files, which, for both the SemEval-2016 and M-ABSA dataset, converts the `.txt` files into `.json` files. If you would prefer to do this manually, please use the `data/xml_to_txt.py` and the `data/basic_data_read.py` files.

If you wish to also run the models on multilingual datasets, please also run the section called "Create Multilingual Datasets", which uses the `data/create_multilingual.py` function. 

If you wish to run fine-tuned LLMs, please create the datasets needed for fine-tuning by also running the section called "Create LLM Finetuning Datasets", which utilises the `data/create_llm_data.py` function.

### mT5-based models

For each these models, we have seperated shells which are split by two categories: "Language" and "Dataset". In `all_shells`, you can hence find the respective file `Language_Dataset_CL_MvP_TASD.py`. Please run "Notebook Setup" girst.

For each of theses languages and datasets, please run the section called "Model Training", which utilises the `model/main_train.py` function to train the various models. This allows for a replication of our results.

Once this has been completed, please run the section called "Model Evaluation: Language". This again uses the `model/main_train.py` function, but for model evaluation. This is achieved by not including parameter `--do_train`, but instead `do_inference`, `do_inference_best_5` or `do_inference_best_ckpt`.

Note that the files for the Multilingual(_Small) and English languages are slightly different. They both have a section called "Model Training", but a "Model Evaluatiom: Language" section for each language from the following list:
    - English
    - Dutch
    - French
    - Spanish
    - Turkish

Lastly, the file `Multilingual_Small_MABSA_CL_MvP_TASD.ipynb` has an entire new section called "Model - Limited Size", which contains an entire section to train and evaluate models on a multilingual dataset that is limited to the same size with respect to sentences as the individual language datasets.

### LLM Models

For all LLM models across datasets, you can use the file `0_LLM_Comparison_ML_CL_MvP_TASD.ipynb`. Again, run the "Notebook Setup". You have to insert your own "OPENAI_API_KEY".

For each "Dataset" from the list \[SemEval-2016, M-ABBSA\], run the section called "Dataset: Text Generation". First, in the subsection "Fine-Tuning", the respective GPT-4.1-mini model is finetuned, using the function `llm/generation/fine_tune.py`. Then, text generation is carried out for all different prompt combinations, for each language, as well as for the fine-tuned models. This uses the function `llm/generation/inf.py`. 

Once the "Dataset: Text Generation" section has run, please run the corresponding "Dataset: Eval" section. Here, the function `llm/generation/eval.py` is used to evaluate the accuracy of the text generation.

### Results - Combine

Once the previous steps have been run, you have results, albeit scattered across files. Lastly, you again used the `0_Data_CL_MvP_TASD.ipynb` file. Once you have run the "Notebook Setup", please run the section "Combine Results". You can also run individual functions here if you have not trained all models. Here `data/results_collation.py` is used to create results files that are easier to read.

## Acknowledgements

Our model is inspired by and partially adapted from code by:

- Peper, J. & Wang, L. (2022). Generative aspect-based sentiment analysis with contrastive learning and expressive structure. In Findings of the association for computational linguistics: (EMNLP, 2022) (pp. 6089–6095). ACL. GitHub: https://github.com/jpeper/GEN_SCL_NAT.git
- Gou, Z., Guo, Q. & Yang, Y. (2023). MvP: Multi-view prompting improves aspect sentiment tuple prediction. In 61st Annual Meeting of the Association for Computational Linguistics (ACL 2023) (pp. 4380–4397). ACL. GitHub: https://github.com/ZubinGou/multi-view-prompting.git
- Wu, C., Ma, B., Liu, Y., Zhang, Z., Deng, N., Li, Y., . . . Xue, Y. (2025). M-ABSA: A multi-lingual dataset for aspect-based sentiment analysis. (arXiv preprint arXiv: 2502.11824). GitHub: https://github.com/swaggy66/eval-llm-mabsa.git


