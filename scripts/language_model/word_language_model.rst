Word Language Model
-------------------

Merity, S., et al. "`Regularizing and optimizing LSTM language models <https://openreview.net/pdf?id=SyyGPP0TZ>`_". ICLR 2018
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:download:`[Download] </scripts/language_model.zip>`

The key features used to reproduce the results for pre-trained models are listed in the following tables.

.. editting URL for the following table: https://bit.ly/2HnC2cn

The dataset used for training the models is wikitext-2.


+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Model        | awd_lstm_lm_1150_wikitext-2     | awd_lstm_lm_600_wikitext-2     | standard_lstm_lm_1500_wikitext-2     | standard_lstm_lm_650_wikitext-2     | standard_lstm_lm_200_wikitext-2     |
+==============+=================================+================================+======================================+=====================================+=====================================+
| Mode         | LSTM                            | LSTM                           | LSTM                                 | LSTM                                | LSTM                                |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Num_layers   | 3                               | 3                              | 2                                    | 2                                   | 2                                   |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Embed size   | 400                             | 200                            | 1500                                 | 650                                 | 200                                 |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Hidden size  | 1150                            | 600                            | 1500                                 | 650                                 | 200                                 |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Dropout      | 0.4                             | 0.2                            | 0.65                                 | 0.5                                 | 0.2                                 |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Dropout_h    | 0.2                             | 0.1                            | 0                                    | 0                                   | 0                                   |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Dropout_i    | 0.65                            | 0.3                            | 0                                    | 0                                   | 0                                   |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Dropout_e    | 0.1                             | 0.05                           | 0                                    | 0                                   | 0                                   |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Weight_drop  | 0.5                             | 0.2                            | 0                                    | 0                                   | 0                                   |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Tied         | True                            | True                           | True                                 | True                                | True                                |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Val PPL      | 73.32                           | 84.61                          | 98.29                                | 98.96                               | 108.25                              |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Test PPL     | 69.74                           | 80.96                          | 92.83                                | 93.90                               | 102.26                              |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+
| Command      | [1]                             | [2]                            | [3]                                  | [4]                                 | [5]                                 |
+--------------+---------------------------------+--------------------------------+--------------------------------------+-------------------------------------+-------------------------------------+

[1] awd_lstm_lm_1150_wikitext-2 (Val PPL 73.32 Test PPL 69.74)

.. code-block:: console

   $ python word_language_model.py --gpus 0 --tied --save awd_lstm_lm_1150_wikitext-2

[2] awd_lstm_lm_600_wikitext-2 (Val PPL 84.61 Test PPL 80.96)

.. code-block:: console

   $ python word_language_model.py -gpus 0 --emsize 200 --nhid 600 --dropout 0.2 --dropout_h 0.1 --dropout_i 0.3 --dropout_e 0.05 --weight_drop 0.2 --tied --save awd_lstm_lm_600_wikitext-2

[3] standard_lstm_lm_1500_wikitext-2 (Val PPL 98.29 Test PPL 92.83)

.. code-block:: console

   $ python word_language_model.py --gpus 0 --emsize 1500 --nhid 1500 --nlayers 2 --lr 20 --epochs 750 --batch_size 20 --bptt 35 --dropout 0.65 --dropout_h 0 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --wd 0 --alpha 0 --beta 0 --save standard_lstm_lm_1500_wikitext-2

[4] standard_lstm_lm_650_wikitext-2 (Val PPL 98.96 Test PPL 93.90)

.. code-block:: console

   $ python word_language_model.py --gpus 0 --emsize 650 --nhid 650 --nlayers 2 --lr 20 --epochs 750 --batch_size 20 --bptt 35 --dropout 0.5 --dropout_h 0 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --wd 0 --alpha 0 --beta 0 --save standard_lstm_lm_650_wikitext-2

[5] standard_lstm_lm_200_wikitext-2 (Val PPL 108.25 Test PPL 102.26)

.. code-block:: console

   $ python word_language_model.py --gpus 0 --emsize 200 --nhid 200 --nlayers 2 --lr 20 --epochs 750 --batch_size 20 --bptt 35 --dropout 0.2 --dropout_h 0 --dropout_i 0 --dropout_e 0 --weight_drop 0 --tied --wd 0 --alpha 0 --beta 0 --save standard_lstm_lm_200_wikitext-2
