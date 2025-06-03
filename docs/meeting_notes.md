# Meeting Notes

## 07.05.2025

### Regularization
- not use 1SE (gives too big lambdas)
- fix lambda for all noise levels and so on
- why is lambda so big?

### Matrix generation
- multiplicative noise on simulated clusters -> works better than additive
    - 1, 2, 3 times +- 10%
- all BRs on 0 give very good results
- increased BRs to behave similar to cMHN base rates 

### Distance measure
- rn only based on IRs
- rn l2 -> maybe try l1

### Thesis

- Start writing introduction
- chapter 3 for evaluation on simulated thetas
- chapter 4 application to real dataset
    - includes desscription of biological pathways

## 14.05.2025

### Regularization

- fixed a t 1/datasetsize

### Distance measure
- compared symmetrized version of thetas (sum with transpose) --> worked very well
- other option: use symsparse instead of l1
- use other test events? Based on their functionalities? -> Andi's list
- use multiple test event groups
- compare stability across different test events

### Application
- Application to LUAD dataset
    - KEAP and STK11 clustered together
    - research functions of all genes
