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

## 06.06.2025

### Update

- calculated maximum distance over 4 distance metrics
- not looked at biological functions of all genes
- similar according to new dist measure: ARID1A, ARID2 and RB1 Mut, Del
- Andi wegen Biologischer Bedeutungen und wegen Funktionsrecherche fragen
- EGFR (M) immer große Distanz
  - hat vor allem starke Distanzen wenn KEAP und STK11 Tesstevents sind

### Thesis

- adding noise Kapitel 
  - Vergleich verschiedener Noise Stärken
  - Supervised Clustering Measure 
    -> Rainer fragen
- in depth analyse für 2 
  - mit Andi absprechen
  - Biopathways anschauen und entsprechende Events in Matrix suchen

### Orga

- Dauer Berechnung Datensatz: ca. 4 Minuten
- Ziel: in 5 Wochen fertig 