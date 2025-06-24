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

## 13.06.2025

- Thesisk
  - wrote MHN part
  - evaluation part
    - Redid new noise: Now not iteratively add multiplicative noise, but add increasing noise
    - rand score:
      - mean/max/min score over different cluster numbers
      - heuristic for custer cut off --> do not need it for the evaluation, bc we know the ground truth

- Application
  - genie16 analysis.ipynb
  - chromatin remodelling: ARID1,2 KMTD2
  - MPAK pathway
  - mean distanz noch nicht ausprobiert --> normalisieren über die Größe der Theta-Matrizen
- next MHN meeting presentation of results
- until next time finish evaluation chapter

## Presentation 16.06.2025

- Noise on data rather than theta matrix (from data preprocessing / data generation)
- results very good, would not expect clusters of pathways
- summe statt max
- Publikation?
- mit Korrelation vergleichen: 
  - stark korrelierte betrachten als Sanity Check
  - stark antikorellierte betrachten und biologisch interpretieren

## 24.06.2025

- mean instead of max
- normalization
  - by number of entries/ off-diags: dominated by small test event numbers (big distance for EGFR, but persists after taking only 3er test events)
  - by number of test events
  - plot norm of a big number of MHNs of sizes 2-4 and see whether linear or quadratic relationship
- test events: 2er, 3er, 4er
- test whether there's a bias from the choice of test event sets -> no
- correlation sanity check
  - some clusters are not simply correlated
  - but events with high correlation appear in same cluster
- wrote 
  - simulation chapter
    - may still do confidence intervals for rand score
  - missing:
    - combination of distances from different test sets
    - application (until next time (?))
    - application to other dataset (Linda will ask Andi for type and test sets)
