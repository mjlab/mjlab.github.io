---
title: Speech Recognition with Sequence to Sequence Models
description: state-of-the-art speech recognition with sequence to sequence
categories:
 - Paper
tags:
 - 머신러닝
 - 딥러닝
 - 음성인식
---

## Information
### Title
* STATE-OF-THE-ART SPEECH RECOGNITION WITH SEQUENCE TO SEQUENCE MODELS

### Author
* Chung-Cheng Chiu

### Affiliation
* Google

### Link
* [pdf](https://arxiv.org/abs/1712.01769)

### 주요 기술
* Word Piece Models [[arxiv](https://arxiv.org/abs/1609.08144)]
* Listen, Attend and Spell (LAS) [[arxiv](https://arxiv.org/abs/1508.01211)]

## Description
### Abstract
* Attention 기반의 encoder-decoder 구조를 제안
  * Listen, Attend, and Spell (LAS)
  * 한 네트워크 안에 acoustic, pronunciation, language model을 포함하는 네트워크

* 이전 논문에서는
  * 받아쓰기 분야 (dictation task)에서 state-of-the-art ASR과 필적하는 결과를 보여줌
  * 음성 검색 (voice search)과 같은 분야에서는 뚜렷한 결과를 보여주지 못함

* 이 논문에서는
  * LAS model의 유의미한 성능 향상을 보여줌
    * 12,500 시간의 음성 검색 부분
    * WER 가 9.6% 에서 5.2%로 상승

  * 구조적인 관점에서 다음의 기술을 사용
    * Grapheme 대신 Wordpiece 모델을 사용함
    * Multi-head attention 구조를 제안

  * 최적화 관점에서 다음의 기술을 사용
    * Synchronous training
    * Scheduled sampling
    * Label smoothing
    * Minimum word error rate

### Introduction
* Sequence-to-sequence 모델의 인기가 늘어감
  * 기존의 ASR 시스템의 모델을 하나의 네트워크로 표현
    * Acoustic model (AM)
    * Pronunciation model (PM)
    * Language model (LM)

  * 다양한 모델이 소개됨
    * Recurrent Neural Netork Tranducer (RNN-T) [[arxiv](https://arxiv.org/abs/1211.3711)]
    * Listen, Attend and Spell (LAS) [[arxiv](https://arxiv.org/abs/1508.01211)]
    * Neural Tranducer [[arxiv](https://arxiv.org/abs/1511.04868)]
    * Monotonic Alignments [[arxiv](https://arxiv.org/abs/1704.00784)]
    * Recurrent Neural Aligner (RNA) [[pdf](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=2&cad=rja&uact=8&ved=0ahUKEwiCkpu_q5bYAhWHkZQKHQgKC3MQFgguMAE&url=https%3A%2F%2Fpdfs.semanticscholar.org%2F7703%2Fa2c5468ecbee5b62c048339a03358ed5fe19.pdf&usg=AOvVaw3v0gJB-eoUBmIdkRcVNVls)]

  * 위에 소개된 모델들은
    * 좋은 성능을 보여줌
    * 기존의 HMM 기반의 시스템을 대체 하기엔 어려움

  * Sequence-to-sequence 모델의 장점으로 다음의 모듈들을 필요로 하지 않음
    * Finite state tranducer (FST)
    * Lexicon
    * Text normalization

* 이 논문의 목적은
  * 음성 검색 부분에서 기존의 ASR system의 성능을 능가
    * 다양한 구조와 최적화를 수행

* 이전 논문에서는 LAS 모델이 다른 RNN-T [[pdf](https://pdfs.semanticscholar.org/6cc6/8e8adf34b580f3f37d1bd267ee701974edde.pdf)] 와 같은 sequence-to-sequence 모델과 비교하여 성능이 향상된 것을 보여줌
* 이 논문에서는 LAS 모델 자체의 성능향상에 초점을 둠
  * LAS 모델은 싱글 네트워크 이며 다음과 같은 구조로 되어있음
    * Encoder는 기존의 acoustic 모델과 비슷한 역할
    * Attender는 alignment 모델과 비슷한 역할
    * Decoder는 기존의 language 모델과 비슷한 역할

  * 구조적인 측면으로
    * Multi-head attention을 결합 [[arxiv](https://arxiv.org/abs/1706.03762)]
      * Encoder feature의 중복 위치를 가능하게 함

    * Word piece models (WPM)을 사용 [[arxiv](https://arxiv.org/abs/1609.08144)]
      * 번역에 적용됨
      * 최근에는 음성에도 적용됨 (RNN-T, LAS)
      * WPM을 사용하여 13% 정도의 상대 성능 개선 효과를 갖음 (WER)

  * 최적화 측면으로
    * Minimum word error rate (MWER)를 사용 [[arxiv](https://arxiv.org/abs/1712.01818)]
    * Scheduled sampling (SS) [[arxiv](https://arxiv.org/abs/1506.03099)]
      * 학습하는 동안 ground truth 대신 이전에 인식한 label을 사용

    * Label smoothing
      * 비전 분야에 적용 [[arxiv](https://arxiv.org/abs/1512.00567)]
      * 음성 분야에 적용 [[arxiv](https://arxiv.org/abs/1506.07503)]

    * 다음의 최적화 방법을 사용
      * Asynchronous training [[pdf](https://research.google.com/archive/large_deep_networks_nips2012.pdf)]
      * Synchronous training [[arvix](https://arxiv.org/abs/1706.02677)]

    * 세가지 방법을 사용한 성능향상
      * 27.5% 의 상대 성능 개선 효과 (WER)

  * 추가적으로
    * Language Model을 이용해 rescoring 을 수행
    * 3.4% 정도의 상대 성능 개선

### System Overview
#### Basic LAS Model
* 시스템 구성도
![시스템 구성도](https://drive.google.com/uc?id=1USwaHTFQRqi2uPiAEcde-MMZ_IgxprCv)

  * Listener는 Encoder 부분을 의미함
    * 기존의 음성인식 방법의 acoustic model 부분
    * 입력특징을 고수준의 특징 $$\mathbf{h}^{enc}$$ 로 변환함

  * Encoder의 출력 (attention context)은 출력 $$\mathbf{h}_{i}$$을 예측하는데 사용됨
    * Dynamic Time Warping (DTW)과 비슷한 역할을 함

  * Speller는 Decoder 부분을 의미함
    * Attender의 출력 (attention context) $$\mathbf{c}_{i}$$를 이용함
    * Sub-word unit $$\mathbf{y}_{i}$$의 확률 분포 $$ P(y_{i} \vert y_{i-1},\cdots,y_{0},x) $$를 예측하기 위해 다음을 사용
      * 입력 $$\mathbf{x}$$
      * 이전에 예측된 sub-word unit $$y_{i-1}$$

#### Structure Improvements
##### Wordpiece models
* 기존의 방법은 AM, PM, LM을 사용
  * Grapheme (characters)를 출력으로 사용
  * 단점으로는 out of vocabulary (OOV)가 발생함

* 대안으로 문맥에 독립적인 음소들(phonemes)을 사용 [[page](https://research.googleblog.com/2017/12/improving-end-to-end-models-for-speech.html)]
  * 기존 방법대로 음소를 사용하면 PM, LM 이 요구됨
  * 이 논문의 실험에서는 성능 향상에 도움이 안됨

* WPM의 장점
  * 일반적인 word-level LM은 grapheme-level LM과 비교하여 perplexity가 낮음
  * Wordpiece가 grapheme보다 강력한 모델이 될 수 있음
  * Longer unit이 LSTM의 메모리를 효율적으로 사용할 수 있게 함
  * Longer unit이 inference 속도를 향상시킴 (매우 유의미)
  * RNN-T와 같은 sequence-to-sequence 모델 보다 WPM이 좋은 성능을 보여줌

* WPM의 특징 [[pdf](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=0ahUKEwjNycXa5JfYAhUIE5QKHWZkDjkQFggmMAA&url=https%3A%2F%2Fresearch.google.com%2Fpubs%2Farchive%2F37842.pdf&usg=AOvVaw2XZCRPXq2ZAssDPHvXoVuv)]
  * WPM의 길이는 grapheme부터 전체 단어까지의 길이를 갖는 sub-word
  * WPM을 사용하면 out-of-vocabulary 문제가 없음
  * 학습셋에 대해 maximum likelihood가 최대가 되는 language model임
  * Word의 boundary는 문맥에 독립적이며 greedy algorithm을 사용하여 결정함

##### Multi-headed attention
* Multi-headed attention 개념도
![개념도](https://drive.google.com/uc?id=165I0bU8WYB0ybcT3fHbbmZ7yVrZjeash)

* Multi-head attention (MHA)은 기계번역에서 처음 사용됨 [[pdf](https://arxiv.org/abs/1706.03762)]
* 이 논문에서는 MHA를 음성에 사용
* 기존의 attention 방법을 multiple head를 갖도록 확장
  * 각각의 head는 encoder의 output에 다른 역할을 하도록 함
  * Decoder가 정보 검색을 쉽게 할 것이라 가정

* 기존의 attention 방법은 attention에서 output을 정확하게 pick하기 위해 명확한 결과를 주어야함
* MHA 방법은 encoder의 부담을 줄여주고 음성과 잡음을 구별짓는 효과가 있다고 가정

#### Optimization Improvements
##### Mimimum Word Error Rate (MWER) Training
* 기존의 ASR system에서는
  * State-level minimum Bayes risk (sMBR) [[pdf](https://pdfs.semanticscholar.org/2443/dc59cf3d6cc1deba6d3220d61664b1a7eada.pdf)]와 같은 sequence level criterion을 최적화함
  * CE, CTC 을 덧붙여 학습
  * 최적화할 metric이 실제 측정할 metric (WER)과 연관이 없음

* 이 논문에서는
  * Minimum word error rate (MWER)에 초점을 맞춤 [[arxiv](https://arxiv.org/abs/1712.01818)]
  * MWER은 word error를 최소화하는 목적 함수를 설정
  * Loss function은 다음과 같음

    $$
      \mathcal{L}_{embr} = \mathbb{E}_{P(y \vert x)} [WordErrors(\mathbf{y}, \mathbf{y}^{ * })] + \lambda \mathcal{L}_{CE}
    $$

    * $$\mathbf{y}$$ : hypothesis
    * $$\mathbf{y^{ * }} $$ : ground-truth label sequence

  * 위의 방법은 다음과 같은 방법으로 근사 가능
    * Sampling [[pdf](https://pdfs.semanticscholar.org/7703/a2c5468ecbee5b62c048339a03358ed5fe19.pdf)]
    * Summation N-best list [[pdf](https://pdfs.semanticscholar.org/2443/dc59cf3d6cc1deba6d3220d61664b1a7eada.pdf)]

  * 이 논문에서는 후자의 방법이 더 효과적임
  * 위의 식은 각각의 결과에 weighted summation으로 근사 가능함

    $$
      \mathcal{L}^{s}_{mwer} = \frac{1}{N} \sum_{y_{i} \in NBest(x, N)} {[WordErrors(\mathbf{y_{i}}, \mathbf{y}^{ * })]} \hat{P}(y_{i} \vert x) + \lambda \mathcal{L}_{CE}
    $$

    * $$ NBest(x, N) = {y_{1}, \cdots, y_{n}} $$ : 입력 $$ \mathbf{x} $$를 beam-search decoder에 의해 계산된 결과 [[arxiv](https://arxiv.org/abs/1409.3215)]
    * $$ \hat{P}(y_{i} \vert x) = \frac{P(y_{i} \vert x)}{\sum_{y_{i} \in NBest(x, N)} P(y_{i} \vert x)} $$ 로 정의된다.

##### Scheduled Sampling
* Decoder를 학습시키는 방법
  * Teacher forcing
    * 이전의 예측값으로 ground-truth label을 사용
    * 초반에 decoder 빠르게 학습시키는데 도움이 됨
    * 학습과 예측에 차이가 발생한다.

  * Scheduled Sampling [[arvix](https://arxiv.org/abs/1506.03099)]
    * 이전 예측의 확률 분포로 부터 샘플링
    * 다음 label을 예측할 때, 결과 token을 이전 token으로 사용

  * 이 논문에서는
    * 학습 시작 시점에는 teacher force 방법을 사용
    * 특정 시점 (모델의 예측 확률이 0.4)이 되면 sampling 방법의 확률을 선형으로 증가시킴
    * 확률이 0.4가 되는 시점
      * Asynchronous : 100만
      * Synchronous : 10만

##### Asynchronous and Synchronous Training
* Asynchronous training [[pdf](https://research.google.com/archive/large_deep_networks_nips2012.pdf)]
* Synchronous training [[arxiv](https://arxiv.org/abs/1706.02677)]
* 두가지 방법 모두 학습 초기에 높은 gradient variance가 문제가 됨
  * Asynchronous 학습은 초기에 모든 replica를 사용하지 않고 점차적으로 늘림
  * Synchronous 학습은 learning rate ramp up과 gradient norm tracker 방법을 사용

##### Label Smoothing
* Label smoothing은 정규화 방법 [[arxiv](https://arxiv.org/abs/1512.00567)]
  * Model이 과적합되지 않도록 수행
  * Ground-truth label의 분포가 uniform distribution이 되도록 함

#### Second-Pass Rescoring
* LAS 모델의 decoder는 language model과 같은 역할을 함
  * 학습 데이터의 transcript에 존재하는 단어만 다룰 수 있음
  * 외부 LM을 사용하는 경우 오디오 데이터가 없는 단어에 대해 예측할 수 있음
  * 외부 LM은 다양한 도메인으로 부터 얻은 텍스트로 학습한 5-gram LM
  * 특정 도메인 LM은 Bayesian-interpolation을 사용 [[pdf](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37567.pdf)]
  * Beam search를 통해 얻은 N-best hypotheses로 부터 다음과 같이 transcript $$ \mathbf{y}^{ * } $$을 정함

    $$
      \mathbf{y}^{ * } = \underset{\mathbf{y}}{\arg \max} log P(\mathbf{y} \vert \mathbf{x}) + \lambda \log P_{LM}(\mathbf{y}) + \gamma len(y)
    $$

    * $$ P_{LM} $$ : 외부 LM의 확률
    * $$ len(y) $$ : $$ \mathbf{y} $$의 단어 수
    * $$ \lambda, \gamma $$ : 학습 셋으로 정해지는 파라미터

### Experimental Details
* Corpus 정보
  * 12,500 시간
  * 15,000,000 영어 발화
  * Noise, Reverberation 추가
    * 0dB ~ 30dB
    * 평균 12dB
    * YouTube, daily life noise environment

  * Feature extraction
    * 80 dimensional log-Mel
    * 25ms window
    * 10ms shift
    * Stacked 3 frame to left

  * Encoder network
    * 5 long short-term memory (LSTM)
    * Unidirectional LSTM [[pdf](http://www.bioinf.jku.at/publications/older/2604.pdf)]
      * 1,400 hidden unit
    * Bidirectional LSTM [[pdf](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.331.9441&rep=rep1&type=pdf)]
      * 2,048 hidden unit (1,024 hidden unit per direction)

  * Attention [[arxiv](https://arxiv.org/abs/1409.0473)]
    * Single-headed attention
    * Multi-headed attention

  * Decoder
    * 2 layer LSTM
      * 1,024 hidden unit per layer

### Results
#### Structure Improvements
* 구조 변화에 따른 성능 변화
  * LAS model + grapheme (E1)
  * LAS model + WPM (E2)
  * LAS model + WPM + MHA (E3)

![표1](https://drive.google.com/uc?id=1qkJiHi6I7ZDen8fZvF7VVcOhfKeZtmzH)

#### Optimization Improvements
* 최적화에 따른 성능 변화
  * E3 + synchronous training (E4)
  * E4 + scheduled sampling (E5)
  * E5 + label smoothing (E6)
  * E6 + MWER training (E7)

![표2](https://drive.google.com/uc?id=1Eslji1uV0QQyGBydr9fPY1rkdjp-6Ksi)

#### Incoroperating Second-Pass Rescoring
* Second-Pass rescoring 을 적용
  * E7 + LM rescoring (E8)
    * 상대 성능으로 3.4% 상승

![표3](https://drive.google.com/uc?id=12SGHYMEbFx0zB0uTWOrCKr7Jq-B0DWYe)

#### Unidirectional vs. Bidirectional Encoders
* 제안한 방법이 모델 구조에 상관없이 성능 향상을 가져옴

![표4](https://drive.google.com/uc?id=1aG2BZ06jaKI4Ji__rgKkV7U3-gMPajpA)

#### Comparison with the Convolutional Systems
* State-of-the-art model [[pdf](https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/45555.pdf)]과 비교하여 성능 향성

![표5](https://drive.google.com/uc?id=1u28RbTkvPEIYouIQTN7i4Kq_tVQKpJoo)

### Conclusion
* 장점
  * AM, PM, LM을 한 네트워크로 만듦
  * lexicon, text normalization 모듈을 필요로 하지 않음

* 한계
  * Unidirectional LAS 시스템은 전체 발화를 얻어야 decode 가능

* Future work
  * Streaming attention-based model을 적용
    * Neural Tranducer
