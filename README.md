# On the Effectiveness of Parameter-Efficient Fine-Tuning
**Zihao Fu**,<sup>1</sup> **Haoran Yang**,<sup>2</sup> **Anthony Man-Cho So**,<sup>2</sup> **Wai Lam**,<sup>2</sup> **Lidong Bing**,<sup>3</sup> **Nigel Collier**<sup>1</sup>



<sup>1</sup>Language Technology Lab, University of Cambridge<br>
<sup>2</sup>The Chinese University of Hong Kong<br>
<sup>3</sup>DAMO Academy, Alibaba Group

## Install
We suggest to create a new conda env to install the dependencies.
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