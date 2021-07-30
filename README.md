## BoB: BERT Over BERT for Training Persona-based Dialogue Models from Limited Personalized Data
[<img src="_static/pytorch-logo.png" width="10%">](https://github.com/pytorch/pytorch) [<img src="https://www.apache.org/img/ASF20thAnniversary.jpg" width="6%">](https://www.apache.org/licenses/LICENSE-2.0)

[<img align="right" src="_static/scir.png" width="20%">](http://ir.hit.edu.cn/)

This repository provides the implementation details for the ACL 2021 main conference paper:

**BoB: BERT Over BERT for Training Persona-based Dialogue Models from Limited Personalized Data**. [[paper]](https://aclanthology.org/2021.acl-long.14/)


## 1. Data Preparation
In this work, we carried out persona-based dialogue generation experiments under a persona-dense scenario (English **PersonaChat**) and a persona-sparse scenario (Chinese **PersonalDialog**), with the assistance of a series of auxiliary inference datasets. Here we summarize the key information of these datasets and provide the links to download these datasets if they are directly accessible.

* **For Persona-Dense Experiments**

	|  Dataset	  | Type  | Language | Usage | Where to Download |
	|  ----  		  | ----  | ----  | ----  | ----  |
	|  ConvAI2 PersonaChat | Dialogue Generation  | English   |  Training | [https://www.aclweb.org/anthology/P18-1205.pdf](https://www.aclweb.org/anthology/P18-1205.pdf) train\_self\_original\_no\_cands & valid\_self\_original\_no\_cands (7801 test dialogues)  |
	|  MNLI | Non-dialogue Inference  | English  | Training  | [https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip](https://cims.nyu.edu/~sbowman/multinli/multinli_1.0.zip) entailment & contradiction  |
	|  DNLI | Dialogue Inference  | English  | Evaluation  | [https://www.aclweb.org/anthology/P19-1363.pdf](https://www.aclweb.org/anthology/P19-1363.pdf) |


* **For Persona-Sparse Experiments**

	|  Dataset	  | Type  | Language | Usage | Where to Download |
	|  ----  		  | ----  | ----  | ----  | ----  |
	|  ECDT2019 PersonalDialog | Dialogue Generation  | Chinese   |  Training |    [https://arxiv.org/pdf/1901.09672.pdf](https://arxiv.org/pdf/1901.09672.pdf) dialogues\_train.json & test\_data\_random.json & test\_data\_biased.json |
	|  CMNLI | Non-dialogue Inference  | Chinese  | Training  | [https://github.com/CLUEbenchmark/CLUECorpus2020/](https://github.com/CLUEbenchmark/CLUECorpus2020/) entailment & contradiction |
	|  KvPI | Dialogue Inference  | Chinese  | Evaluation  | [https://github.com/songhaoyu/KvPI](https://github.com/songhaoyu/KvPI) |
	
	
* **Download Pre-trained BERT**

	The BoB model is initialized from public BERT checkpoints:
 
	* **English BERT**: [https://huggingface.co/bert-base-uncased/tree/main](https://huggingface.co/bert-base-uncased/tree/main)
	* **Chinese BERT**: [https://huggingface.co/bert-base-chinese/tree/main](https://huggingface.co/bert-base-chinese/tree/main)

## 2. How to Run

The `setup.sh` script contains the necessary dependencies to run this project. Simply run `./setup.sh` would install these dependencies. Here
we take the English PersonaChat dataset as an example to illustrate how to run the dialogue generation experiments. Generally, there are three steps, i.e., **tokenization**, **training** and **inference**:

* **Preprocessing**

	```
	python preprocess.py --dataset_type convai2 \
	--trainset ./data/ConvAI2/train_self_original_no_cands.txt \
	--testset ./data/ConvAI2/valid_self_original_no_cands.txt \
	--nliset ./data/ConvAI2/ \
	--encoder_model_name_or_path ./pretrained_models/bert/bert-base-uncased/ \
	--max_source_length 64 \
	--max_target_length 32
	```
	We have provided some data examples (dozens of lines) at the `./data` directory to show the data format. `preprocess.py` reads different datasets and tokenizes the raw data into a series of vocab IDs to facilitate model training. The `--dataset_type` could be either `convai2` (for English PersonaChat) or `ecdt2019` (for Chinese PersonalDialog). Finally, the tokenized data will be saved as a series of JSON files.

* **Model Training**

	```
	CUDA_VISIBLE_DEVICES=0 python bertoverbert.py --do_train \
	--encoder_model ./pretrained_models/bert/bert-base-uncased/ \
	--decoder_model ./pretrained_models/bert/bert-base-uncased/ \
	--decoder2_model ./pretrained_models/bert/bert-base-uncased/ \
	--save_model_path checkpoints/ConvAI2/bertoverbert --dataset_type convai2 \
	--dumped_token ./data/ConvAI2/convai2_tokenized/ \
	--learning_rate 7e-6 \
	--batch_size 32
	```
	
	Here we initialize encoder and both decoders from the same downloaded BERT checkpoint. And more parameter settings could be found at `bertoverbert.py`.
	
* **Evaluations**

	```
	CUDA_VISIBLE_DEVICES=0 python bertoverbert.py --dumped_token ./data/ConvAI2/convai2_tokenized/ \
	--dataset_type convai2 \
	--encoder_model ./pretrained_models/bert/bert-base-uncased/  \
	--do_evaluation --do_predict \
	--eval_epoch 7
	```
	
	Empirically, in the PersonaChat experiment with default hyperparameter settings, the best-performing checkpoint should be found between epoch 5 and epoch 9. If the training procedure goes fine, there should be some results like:
	
	```
	Perplexity on test set is 21.037 and 7.813.
	```
	where `21.037` is the ppl from the first decoder and `7.813` is the final ppl from the second decoder. And the generated results is redirected to `test_result.tsv`, here is a generated example from the above checkpoint:
	
	```
	persona:i'm terrified of scorpions. i am employed by the us postal service. i've a german shepherd named barnaby. my father drove a car for nascar.
	query:sorry to hear that. my dad is an army soldier.
	gold:i thank him for his service.
	response_from_d1:that's cool. i'm a train driver.
	response_from_d2:that's cool. i'm a bit of a canadian who works for america.  
	```
	where `d1` and `d2` are the two BERT decoders, respectively.
	

* **Computing Infrastructure:**
	* The released codes were tested on **NVIDIA Tesla V100 32G** and **NVIDIA PCIe A100 40G** GPUs. Notice that with a `batch_size=32`, the BoB model will need at least 20Gb GPU resources for training.



## MISC
* Build upon 🤗 [Transformers](https://github.com/huggingface/transformers).

* Bibtex:

	<pre>
	@inproceedings{song-etal-2021-bob,
	    title = "{B}o{B}: {BERT} Over {BERT} for Training Persona-based Dialogue Models from Limited Personalized Data",
	    author = "Song, Haoyu  and
	      Wang, Yan  and
	      Zhang, Kaiyan  and
	      Zhang, Wei-Nan  and
	      Liu, Ting",
	    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
	    month = aug,
	    year = "2021",
	    address = "Online",
	    publisher = "Association for Computational Linguistics",
	    url = "https://aclanthology.org/2021.acl-long.14",
	    doi = "10.18653/v1/2021.acl-long.14",
	    pages = "167--177",
	}
	</pre>

* Email: *hysong@ir.hit.edu.cn*.
