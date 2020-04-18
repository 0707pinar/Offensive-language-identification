# Offensive Language Identification for Turkish Language
[System Description]
Hereby I describe  a  system  (pin_cod_) built for SemEval 2020 Task 12: OffensEval: Multilingual Offensive Language Identification in Social Media (Zampieri et al., 2020). The pin_cod_ system is one of participants of OffensEval 2020 Sub-task A: Offensive language  identification for Turkish language (Çöltekin,  2020). 

## Goal:
* Build a model to predict offensive language for each Turkish tweet provided by the organizers of SemEval 2020.
* To predict the labels for the given test set consisting of 3528 unlabeled tweets, the provided training set containing 31756 labeled tweets
* Binary classification task

## Classes to be predicted:
* NOT: not offensive
* OFF: offensive

## Language:
* Turkish

## Model:
* based on bidirectional  long  short-term memory networks (Hochreiter and Schmidhuber,1997),
* incorporated with various lexicon based features and the presence of user mentions (e.g.  @username), 
* followed by two fully connected dense layers.  

## Added files:
* ####pin_cod_preprocessing.py: This includes all preprocessing steps and features to be used for both training and test sets provided by the organizers of SemEval 2020.

* ####pin_cod_model.py: This script is created to build the model (Bidirectional Long Short-Term Memory Networks) automatically predicting offensive language for the provided Turkish tweets.

## Datasets and Resources:
* Training and Test sets were provided by the organizer (Çöltekin,  2020). 
* Turkish  fastText  embeddings (Joulin et al., 2018) was employed in the BiLSTM model (https://github.com/facebookresearch/fastText)
* NRC Emotion Lexicon also  called  EmoLex  (Mohammad  and  Turney, 2013) 
* HurtLex  (Bassignana  et  al.,  2018) which is a  lexicon  consisting  of  negative  stereotypes,  hate  words,  slurs  beyond  stereotypes and  other  words  and  insults,  was used for  Turkish.
*Various wordlists were compiled:
    * https://en.wikibooks.org/wiki/Turkish/Slang#Offensive
    * https://github.com/ooguz/turkce-kufur-karaliste/blob/master/README.md
    * http://tanersezer.com/?p=239

## References:
(1) Elisa  Bassignana,  Valerio  Basile,  and  Viviana  Patti.2018.  Hurtlex:  A multilingual lexicon of words to hurt. In 5th Italian Conference on Computational Linguistics, CLiC-it 2018, volume 2253, pages 1–6.CEUR-WS.

(2) Çağrı Çöltekin. 2020. A Corpus of Turkish Offensive Language on Social Media. In Proceedings of the 12th International Conference on Language Resources and Evaluation. ELRA.

(3) Sepp   Hochreiter   and   Jurgen   Schmidhuber. 1997.Long  short-term  memory.Neural  computation,9(8):1735–1780.

(4) Armand  Joulin,  Piotr  Bojanowski,  Tomas  Mikolov, Herve  Jegou,  and  Edouard  Grave.  2018.   Loss  in translation: Learning bilingual word mapping with a retrieval criterion.  In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing.

(5) Saif M Mohammad and Peter D Turney. 2013. Crowdsourcing a word–emotion association lexicon. Computational Intelligence, 29(3):436–465.

(6) Marcos Zampieri, Preslav Nakov, Sara Rosenthal, Pepa Atanasova, Georgi Karadzhov, Hamdy Mubarak, Leon Derczynski, Zeses Pitenis, and Cağrı Çöltekin. 2020. SemEval-2020 Task 12: Multilingual Offensive Language Identification in Social Media (OffensEval 2020). In Proceedings of SemEval.
