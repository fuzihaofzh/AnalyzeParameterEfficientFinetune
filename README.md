# <div align="center">On the Effectiveness of Parameter-Efficient Fine-Tuning</div>
<div align="center"><b>Zihao Fu,<sup>1</sup> Haoran Yang,<sup>2</sup> Anthony Man-Cho So,<sup>2</sup> Wai Lam,<sup>2</sup> Lidong Bing,<sup>3</sup> Nigel Collier<sup>1</sup></b></div>


<div align="center">
<sup>1</sup>Language Technology Lab, University of Cambridge<br>
<sup>2</sup>The Chinese University of Hong Kong<br>
<sup>3</sup>DAMO Academy, Alibaba Group
</div>

[[Paper (Full+Appendix)]](https://arxiv.org/pdf/2211.15583.pdf)
[[Slides]](https://github.com/fuzihaofzh/AnalyzeParameterEfficientFinetune/blob/main/slides.pdf)


## Takeaways
- This paper gives a comprehensive explanation of why parameter-efficient models (such as Adapters, LoRA, Bitfit, etc.) achieve promising results.
- This paper unveils how the sparsity itself improves the model stability and generalization capability theoretically and empirically.
- This paper proposes a provable approximately best method to choose the tunable parameters for parameter-efficient models.

## Install
This code is the SAM model proposed in the paper. We suggest to create a new conda env to install the dependencies.
```
git clone https://github.com/fuzihaofzh/AnalyzeParameterEfficientFinetune.git
cd AnalyzeParameterEfficientFinetune 
./scripts/install.sh
```

## Run
Run the following code to train our SAM model on the CoLA dataset.
```
./scripts/train.sh
```
