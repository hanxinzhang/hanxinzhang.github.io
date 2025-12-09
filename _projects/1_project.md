---
layout: page
title: "Digital mirror universe: advancing the digital twin framework for multiscale modeling"
description: 
img: assets/img/12.jpg
importance: 1
category:
related_publications: false
---

## Using EHR claims data as an example  

<br>

### Introduction

A digital twin in healthcare research is a dynamic virtual model of an individual patient or biological system, created by integrating clinical, biological, and sensor data with computational simulations. Continuously updated to reflect a patient’s real-time state, it allows researchers to model disease progression, test interventions in silico, and predict treatment responses. This approach supports precision medicine by enabling more personalized and data-driven healthcare decisions.

An electronic health record (EHR) systematically documents a patient’s diagnostic history, procedures, prescriptions, and other clinical information, reflecting their health status as it evolves over time and across care settings. This study aims to develop a model that encodes a subset of a patient's EHR, represented by diagnosis codes (claims), into a stable, time-invariant vector embedding. This embedding serves as a latent digital representation of the patient's underlying health state. Collectively, these embeddings form a structured space that represents the broader patient population, extending the concept of a digital twin. We refer to this as a **digital mirror universe**: a virtual representation of the entire population that can be used for multiple tasks, including disease risk prediction, missing-data imputation, synthetic EHR generation, and downstream integration with genetic and other biomedical data.

A diagnosis EHR, which we refer to as a "sentence", is composed of tens of repeating segment (each a "word" of the form ```Diagnosis_code:Age:Time```). It is worth noting several key characteristics of this medical code sentence:

1. The length or number of segments is not fixed, as in most language models, but the variance may be larger. A large portion of the population has only one or a few diagnosis records, whereas some individuals have hundreds or even thousands. In our sample dataset, the typical sequence length ranges from about 1 to 100 words.

2. The start and end of a sentence (```<SOS>```, ```<EOS>```) are not as clearly defined as in natural language modeling because we do not have complete life-long records for each person. Individuals may be enrolled in the data for only a few weeks, months, or years. We use their enrollment dates, when available, as tentative markers for ```<SOS>``` and ```<EOS>```.

3. Diagnoses occur at irregular time intervals, and there is no reason to expect a fixed window between two contiguous words (medical diagnosis codes). Therefore, commonly used language-modeling techniques such as recurrent neural networks (RNNs) and their variants must be adapted to accommodate this irregular temporal structure. Prior approaches include neural ordinary differential equations [^1], time-dependent embeddings [^2], and explicit positional encodings using large-period Fourier features [^3].

4. The words in a sentence are loosely correlated, especially for EHR data with only a few words. In many cases, people may get sick in a random way. A patient may have infections in their childhood and cardiovascular diseases in their old age, and these two groups of diagnoses may or may not have any correlation. This is very different from modeling natural language, where words in a sentence are expected to be more strongly coupled, grammatically or logically.

5. Some covariates or confounding factors, such as sex, geographic location, and other environmental variables, may also influence the sentence embedding. Depending on the target task, we may or may not want a vector representation that incorporates such confounding information. In the current setting, we aim to encode a patient's EHR into a sex-irrelevant representation, meaning that the embedding should capture the patient's underlying health state but not their sex. Furthermore, from this representation, we should still be able to reconstruct the patient's diagnostic history while treating sex as a covariate.

Mathematically, the goal of this project is to learn a pair of encoder $f$ and decoder $g$, and embed patients' EHRs into a latent space representing their intrinsic health states:

$$
\begin{align}
z &= f(y), \\
\hat y &= g(z),
\end{align}
$$

where $y \sim \mathrm{Pop}$ is a sample of an EHR diagnosis sentence. $\mathrm{Pop}$ denotes the distribution of the patient population. With this latent variable $z$, we aim to reconstruct, predict, impute, and support downstream tasks. We also aim to shape the distribution of $z$ so that it is well-structured for efficient sampling. The resulting latent variable space $z \sim \mathrm{Lat}$ is what we referred as a **digital mirror universe**, which contains the information of the patient population and captures its characteristics.

### Data preparation

The dataset used in this study comes from the IBM MarketScan research database family [^4]. It contains diagnosis histories for more than 150 million unique U.S. patients, collected and manually curated by several large insurers. Diagnoses are recorded using ICD (International Classification of Diseases) codes, which we mapped into roughly 600 consolidated conditions and diseases for simplicity of the vocabulary. For the current training phase, we used 100 million patients whose EHR ‘sentence’ lengths range from 1 to 100, due to limitations in model capacity—long and loosely correlated sequences are difficult for the network to learn effectively. We held out 50,000 patients for validation and model tuning, and reserved another 50,000 for final testing.

### Model 0: Autoencoder

A naive baseline for the task described above is a conditional autoencoder (AE):

$$
\begin{align}
\Phi^*, \Psi^* &= \mathop{\arg \min}\limits_{\Phi, \Psi} \mathbb{E}_{x, y \sim \mathrm{Pop}}\mathbb{E}_{z \sim P_{\Phi}(z|y, x)} -\log P_{\Psi}(y|z, x).
\end{align}
$$

Here, $y$ denotes an EHR diagnosis sentence, $x$ represents a confounding variable (e.g., sex), and $z$ is the latent representation of the EHR. The parameters $\Phi$ and $\Psi$ correspond to the encoder and decoder, respectively. Figure 1 illustrates the architecture of the autoencoder model. The multi-head gated recurrent units (GRUs) consist of several independent GRUs [^5], whose outputs are concatenated through a final aggregation layer. Similar design principles appear in bidirectional recurrent neural networks (RNNs) [^6], where two independent RNNs operating in opposite directions are stacked and their hidden states concatenated. Likewise, the Transformer model [^7] adopts a multi-head structure in which multiple self-attention heads are combined via concatenation.

<p style="text-align: center;">
    <img src="/assets/img/AE_Model_cropped.pdf" width="400">
</p>
<p style="text-align: center;">
    <em>Figure 1. The structure of the autoencoder model. Both the encoder and the decoder are built with multi-head GRUs. The output of all heads of the encoder are concatenated into the latent variable $z$, and passed to decoder, head to head, to generate the conditional distribution $P(y|z, x)$ for loss computation.</em>
</p>

### References

[^1]: Chen, T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. (2018). Neural ordinary differential equations. In *Advances in Neural Information Pro-cessing Systems*, pages 6572–6583.

[^2]: Li, Y., Du, N., and Bengio, S. (2017). Time-dependent representation for neural event sequence prediction. *arXiv preprint arXiv:1708.00065.*

[^3]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems*, pages 5998–6008.

[^4]: IBM Watson Health (2019). IBM MarketScan Research Databases

[^5]: Cho, K., Van Merri¨enboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. (2014). Learning phrase representations using rnn encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078.*

[^6]: Schuster, M. and Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11):2673–2681.

[^7]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems*, pages 5998–6008.
