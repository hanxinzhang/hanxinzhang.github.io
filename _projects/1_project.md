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

<div style="text-align: center;">
  December, 2025 @ Pittsburgh, PA
</div>

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
\Phi^*, \Psi^* &= \mathop{\arg \min}\limits_{\Phi, \Psi} \ \mathbb{E}_{x, y \sim \mathrm{Pop}}\mathbb{E}_{z \sim P_{\Phi}(z|y, x)} -\log P_{\Psi}(y|z, x).
\end{align}
$$

Here, $y$ denotes an EHR diagnosis sentence, $x$ represents a confounding variable (e.g., sex), and $z$ is the latent representation of the EHR. The parameters $\Phi$ and $\Psi$ correspond to the encoder and decoder, respectively. Figure 1 illustrates the architecture of the autoencoder model. The multi-head gated recurrent units (GRUs) consist of several independent GRUs [^5], whose outputs are concatenated through a final aggregation layer. Similar design principles appear in bidirectional recurrent neural networks (RNNs) [^6], where two independent RNNs operating in opposite directions are stacked and their hidden states concatenated. Likewise, the Transformer model [^3] adopts a multi-head structure in which multiple self-attention heads are combined via concatenation.

<p style="text-align: center;">
    <img src="/assets/img/AE_Model_cropped.pdf" width="600px">
</p>

<div style="text-align: center;">
  <div style="display: inline-block; text-align: left; width: 600px;">
      Figure 1. Structure of the autoencoder model. Both the encoder and the decoder are constructed using multi-head GRUs. The outputs from all encoder heads are concatenated to form the latent variable $z$, which is then passed head-to-head to the decoder to generate the conditional distribution $P(y|z, x)$ for loss computation.
  </div>
</div>

<br>

A sample $y$ is fed into the encoder in reverse order, preceded by an ```<EOS>``` token. For each token ```Diagnosis_code:Age:Time```, we embed the diagnosis–age and time components separately using lookup embedding layers. Specifically, a condition observed within a particular age group is treated as a single diagnosis–age token. We consider 605 conditions and discretize age into 24 groups (0-4, 5–9, ..., 115–119), together with ```<SOS>```, ```<EOS>```, and ```<PAD>```, yielding a total vocabulary size of 14,523 for the diagnosis–age embedding layer. Embedding diagnosis and age jointly is motivated by the fact that the same disease occurring at different ages may differ substantially in etiology, pathology, and epidemiology -- for example, early-onset versus late-onset Alzheimer's disease. Time is embedded by treating the week number, relative to the start of the dataset, as a categorical variable and mapping it through a separate lookup embedding table.

To decode the latent variable $z$, we start from the embedded ```<SOS>``` and generate words recurrently. A confounding factor $x$ is fed to both the encoder and the decoder as a conditional variable. We expect the latent variable $z$ to be $x$-irrelevant. In the current setting, we only include sex as a confounding factor, coded by $−1$ and $1$ for males and females.

### Model 1: Wasserstein Autoencoder

An autoencoder is an outstanding architecture for learning a compressed representation. However, there is little constraint imposed on the latent space, so a high-performance autoencoder can degenerate into a copy network that simply learns an identity function:

$$
\begin{align}
y &= f \circ g (y) = f \circ f^{-1} (y),
\end{align}
$$

where the decoder is the inverse of the encoder. There exist many arbitrary choices of latent spaces that can embed our EHR sequences, and there is no natural way to sample from the latent space. Variational autoencoders (VAE) [^7] are known to address this issue by not only encoding the input but also disentangling and embedding it into a well-structured latent space:

$$
\begin{align}
\Phi^*, \Psi^* &= \mathop{\arg \min}\limits_{\Phi, \Psi} \ \mathbb{E}_{x, y \sim \mathrm{Pop}}\left[\mathcal{L}_{\mathrm{reg}} + \mathcal{L}_{\mathrm{ae}}\right] \\
&= \mathop{\arg \min}\limits_{\Phi, \Psi} \ \mathbb{E}_{x, y \sim \mathrm{Pop}} \left[\mathrm{KL}(P_{\Phi}(z|x,y), P(z)) + \mathbb{E}_{z \sim P_{\Phi}(z|y, x)} -\log P_{\Psi}(y|z, x) \right],
\end{align}
$$

where $$\mathcal{L}_{\mathrm{reg}}$$ is the regularization loss and $$\mathcal{L}_{\mathrm{ae}}$$ is the autoencoder reconstruction loss. Intuitively, the regularizer pulls the latent space toward the target prior $P(z)$ -- typically an isotropic Gaussian -- while the autoencoder term attempts to reconstruct the input from the latent embedding. These two objectives can conflict with each other and may induce failure modes or poor reconstruction [^8] [^9].

Tolstikhin *et al.* proposed the Wasserstein Autoencoder (WAE) [^9] to improve upon the variational autoencoder architecture while preserving many of its desirable properties, including strong reconstruction ability, generative modeling capability, stable training, and a disentangled and interpretable latent space. Unlike the vanilla VAE, the WAE explicitly encourages the continuous marginal distribution

$$
\begin{align}
P_{\Phi}(z) = \int P_{\Phi}(z|x,y) d \mathrm{Pop}(x,y) 
\end{align}
$$

to match the prior $P(z)$:

$$
\begin{align}
\Phi^*, \Psi^* &= \mathop{\arg \min}\limits_{\Phi, \Psi} \ \mathbb{E}_{x, y \sim \mathrm{Pop}}\left[\mathcal{L}_{\mathrm{reg}} + \mathcal{L}_{\mathrm{ae}}\right] \\
&= \mathop{\arg \min}\limits_{\Phi, \Psi} \ \mathbb{E}_{x, y \sim \mathrm{Pop}} \left[\lambda \mathcal{D}(P_{\Phi}(z), P(z)) + \mathbb{E}_{z \sim P_{\Phi}(z|y, x)} -\log P_{\Psi}(y|z, x) \right],
\end{align}
$$

where $$\mathcal{D}$$ is an arbitrary divergence and $\lambda > 0$ is the penalty scale. The original paper adopts a generative adversarial network (GAN)-based divergence $$ \mathcal{D}(P_{\Phi}(z), P(z)) =  \mathcal{D}_{\mathrm{JS}}(P_{\Phi}(z), P(z))$$ and uses adversarial training to approximate it. In the present project, we instead employ the Wasserstein-1 metric to overcome the well-known training instabilities of vanilla GAN. The object becomes

$$
\begin{align}
\Phi^*, \Psi^* &= \mathop{\arg \min}\limits_{\Phi, \Psi} \ \mathbb{E}_{x, y \sim \mathrm{Pop}}\left[\mathcal{L}_{\mathrm{reg}} + \mathcal{L}_{\mathrm{ae}}\right] \\
&= \mathop{\arg \min}\limits_{\Phi, \Psi} \ \mathbb{E}_{x, y \sim \mathrm{Pop}} \left[\lambda \mathcal{D_{\mathrm{W}}}(P_{\Phi}(z), P(z)) + \mathbb{E}_{z \sim P_{\Phi}(z|y, x)} -\log P_{\Psi}(y|z, x) \right], \\
\mathcal{D_{\mathrm{W}}}(P_{\Phi}(z), P(z)) &= \frac{1}{K} \sup_{||h||_{L} < K}\mathbb{E}_{z\sim P(z)} h(z) - \mathbb{E}_{z\sim P_{\Phi}(z)} h(z),
\end{align}
$$

where $h$ is a discriminator that facilitates the estimation of the Wasserstein-1 metric. The discriminator is pruned to be $K$-Lipschitz via weight clamping, as in the Wasserstein GAN framework [^10].

The architecture of the model is shown in Figure 2. We append a shallow convolutional neural network (CNN) followed by a linear layer after the encoder for two purposes: (1) to disentangle the latent variable for easier embedding into the target space, and (2) to further compress the latent representation. The encoder together with the CNN and linear layers can be viewed as a generator that aims to produce an isotropic Gaussian latent variable $z$. During training, we optimize three loss terms:

$$
\begin{align}
\mathcal{L}_{\mathrm{ae}} &= \mathbb{E}_{x, y \sim \mathrm{Batch}}\mathbb{E}_{z \sim P_{\Phi}(z|y, x)} -\log P_{\Psi}(y|z, x)\\
\mathcal{L}_{\mathrm{g}} &= -\lambda\mathbb{E}_{x, y \sim \mathrm{Batch}}\mathbb{E}_{z\sim P_{\Phi}(z|y, x)} h(z)\\
\mathcal{L}_{\mathrm{d}} &= \lambda\mathbb{E}_{x, y \sim \mathrm{Batch}}\mathbb{E}_{z\sim P_{\Phi}(z|y, x)} h(z) - \mathbb{E}_{z\sim P(z)} h(z)
\end{align}
$$

where $P(z)$, the prior or target distribution, is chosen to be an isotropic Gaussian $\mathcal{N}(0, \mathrm{I})$.

<p style="text-align: center;">
    <img src="/assets/img/WAE_Model_cropped.pdf" width="800px">
</p>

<div style="text-align: center;">
  <div style="display: inline-block; text-align: left; width: 800px;">
      Figure 2. The Wasserstein autoencoder employs an adversarial discriminator to estimate and reduce the distance between the learned latent space and the target distribution. In our setting, the target distribution is chosen to be an isotropic Gaussian with identity covariance.
  </div>
</div>

### Training and preliminary results

We design four tasks -- reconstruction, interpolation, extrapolation, and generation -- to evaluate the capability of the model (Figure 3). The reconstruction task assesses what proportion of information, including both diagnosis–age and time, can be recovered from the latent variable $z$. Note that in Figure 3, diagnosis–age refers to a diagnosis at an exact age; however, in practice, we group ages into 5-year intervals to reduce the vocabulary size and simplify computation.

For the interpolation task, a subset of diasease–age codes within the input sequence is randomly masked (replaced by ```<PAD>```), and we evaluate the model's ability to impute the missing codes from the latent representation. The extrapolation task is similar, except that only the last few diagnosis–age codes at the end of the sequence are masked.

Finally, if the model successfully embeds sequences into the target latent space (e.g., an isotropic Gaussian), we can directly generate fully synthetic sequences by sampling random Gaussian $z$ and passing it through the decoder.

The training stage is self-supervised (Figure 3). For sequences with length greater than 10, a fixed proportion (currently 10%) of disease–age codes at both the end and within the sequence are randomly masked in the input to facilitate training.

<p style="text-align: center;">
    <img src="/assets/img/task_scheme_cropped2.pdf" width="800px">
</p>

<div style="text-align: center;">
  <div style="display: inline-block; text-align: left; width: 800px;">
      Figure 3. The scheme of training phase and desired tasks.
  </div>
</div>

### References

[^1]: Chen, T. Q., Rubanova, Y., Bettencourt, J., and Duvenaud, D. K. (2018). Neural ordinary differential equations. In *Advances in Neural Information Pro-cessing Systems*, pages 6572–6583.

[^2]: Li, Y., Du, N., and Bengio, S. (2017). Time-dependent representation for neural event sequence prediction. *arXiv preprint arXiv:1708.00065.*

[^3]: Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., and Polosukhin, I. (2017). Attention is all you need. In *Advances in Neural Information Processing Systems*, pages 5998–6008.

[^4]: IBM Watson Health (2019). IBM MarketScan Research Databases

[^5]: Cho, K., Van Merrienboer, B., Gulcehre, C., Bahdanau, D., Bougares, F., Schwenk, H., and Bengio, Y. (2014). Learning phrase representations using rnn encoder-decoder for statistical machine translation. *arXiv preprint arXiv:1406.1078.*

[^6]: Schuster, M. and Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11):2673–2681.

[^7]: Kingma, D. P. and Welling, M. (2013). Auto-encoding varia-tional bayes. *arXiv preprint arXiv:1312.6114.*

[^8]: Zhao, S., Song, J., and Ermon, S. (2017). Infovae: Information maxi-mizing variational autoencoders. *arXiv preprint arXiv:1706.02262.*

[^9]: Tolstikhin, I., Bousquet, O., Gelly, S., and Schoelkopf, B. (2017). Wasserstein auto-encoders. *arXiv preprint arXiv:1711.01558.*

[^10]: Arjovsky, M., Chintala, S., and Bottou, L. (2017). Wasserstein GAN. *arXiv preprint arXiv:1701.07875.*

