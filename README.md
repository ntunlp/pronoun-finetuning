# Pronoun Targeted Fine-tuning for NMT with Hybrid Losses
This repository contains the data and source code of our paper "Pronoun Targeted Fine-tuning for NMT with Hybrid Losses" in EMNLP 2020.


### Prerequisites

```
* PyTorch 0.4 or higher
* Python 3
* fairseq
```


## How To Run
The implementations of the hybrid losses, combining the existing cross-entropy loss implementation in `fairseq` and the additional discriminative losses, are provided here.

* max_margin_loss = CE + sentence-level max-margin/pairwise loss applied to all tokens
* loglikelihood_loss = CE + sentence-level log-likelihood loss applied to all tokens
* pronoun_maxmargin_loss = CE + sentence-level max-margin/pairwise loss applied only to pronouns
* pronoun_loglikelihood_loss = CE + sentence-level log-likelihood loss applied only to pronouns

Place the provided `.py` files defining the new losses in the `fairseq/fairseq/criterion` folder. 

The losses that are only applied to pronouns require a dictionary specifying the indices of the pronouns. The code `get_pronoun_idx.py` can help you get this (for English; you can modify it to list pronouns for another language). Note that this code automatically adds 4 to the index to account for the `<eos>`, `<unk>`, etc. tags. 

After preprocessing your training data, run this code with the `dict.xx.txt` file produced by `fairseq-preprocess` (inside the binary file folder) as input. Make sure to run this with the target dictionary file unless the pre-processing included --joined-dictionary. 

```
> python get_pronoun_idx.py [dict.xx.txt]
```

This should produce two pickle files containing dictionaries, `pronoun_to_index.pkl` and `index_to_pronoun.pkl`. Place these files in the same folder as the loss `(fairseq/fairseq/criterion)`.

You should now be able to train an NMT model using these losses in the regular way with `fairseq`. To specify the criterion on the command line while running `fairseq_cli/train.py`, add `label_smoothed_cross_entropy_` before the loss you want to use. For e.g.,

```
> fairseq_cli/train_.py ... --criterion label_smoothed_cross_entropy_loglikelihood_loss ...
> fairseq_cli/train_.py ... --criterion label_smoothed_cross_entropy_pronoun_maxmargin_loss ...
```
You can make changes to these codes to alter any hyperparameters you may wish to change (e.g. temperature, margin, etc). The default values are the same as specified in the paper.



## Citation
Please cite our paper if you found the resources in this repository useful.
```
@inproceedings{PronounTargetedFinetuning,
  title={Pronoun Targeted Fine-tuning for NMT with Hybrid Losses},
  author={Prathyusha Jwalapuram and Shafiq Joty and Youlin Shen},
  booktitle={EMNLP},
  year={2020}

}	
```
