---
title: RSSI-Based Passenger Movement Classification
subtitle: Non-Intrusive Public Transport Monitoring
author: "\\underline{André Ribeiro}, Guilherme Matos, Julio Corona, Mário Antunes, Diogo Gomes"
aspectratio: 169
institute: Instituto de Telecomunicações, Universidade de Aveiro, Portugal
date: 03/06/206 - EuCNC26
section-titles: false
# toc: true
# toc-title: "Table of Contents"
header-includes:
  - \usetheme[sectionpage=none,numbering=fraction,progressbar=frametitle]{metropolis}
  - \setbeamertemplate{navigation symbols}{}
  - \usepackage{longtable,booktabs}
  - \usepackage[table]{xcolor}
  - \usepackage{etoolbox}
  - \AtBeginSection{}
  - \AtBeginSubsection{}
---

# inMotion

## A Smarter Public Transport

::: columns
:::: {.column width="52%"}

The **inMotion** project aims to bring real-time intelligence to public transport operations.

One of its core goals: understand **how passengers move** through the network: who boards, who alights, and where.


Today we present a component of that vision: passenger movement classification using only Wi-Fi signals.

::::
:::: {.column width="44%"}

\centering
\includegraphics[width=0.85\linewidth]{./images/experimental_setup.png}
\small\color{gray}Experimental prototype at Instituto de Telecomunicações

::::
:::

::: notes

- Welcome everyone. This work is part of the inMotion project, funded by CENTRO 2030.
- The project's mission is to bring intelligence to public transport operations.
- Today I'll focus on one specific challenge: how do we know how many people board and alight at each stop, without installing expensive hardware on every bus?
- The key insight: we use Wi-Fi signals that are already there.

:::

# Why This Matters

## The Passenger Counting Gap


**What operators need:**

- Accurate passenger flow data for scheduling, fleet sizing, and route planning
- Coverage across the entire fleet, not just a few instrumented vehicles
- Low deployment and maintenance cost

::: notes

- Accurate passenger counting is fundamental for public transport operations. Without it, operators are flying blind.
- Traditional systems, infrared beams, pressure mats, cameras, work reasonably well in labs. In the real world, with crowds, occlusion, and dirt, their accuracy drops dramatically — 98% claimed becomes 53% in practice, as Pronello and Ruiz documented.
- Video-based counting is more robust but introduces privacy issues and still requires cameras on every door.
- The core problem: coverage. You cannot instrument an entire fleet with these technologies at a reasonable cost.
- Wi-Fi, however, is already on many buses. The infrastructure is there.

:::

# The Problem

## Four Movements to Classify

::: columns
:::: {.column width="52%"}

We want to answer a simple question for every passenger interacting with the bus door:

**Did this person board, alight, or stay put?**

| Label | Meaning           | Description               |
| ----- | ----------------- | ------------------------- |
| AA    | A $\rightarrow$ A | Remaining inside the bus  |
| BB    | B $\rightarrow$ B | Remaining at the bus stop |
| BA    | B $\rightarrow$ A | **Boarding** the bus      |
| AB    | A $\rightarrow$ B | **Alighting** the bus     |

\footnotesize Zone A = bus interior (near access point) 

\footnotesize Zone B = bus stop (outside)

::::
:::: {.column width="48%"}

\centering
\includegraphics[width=\linewidth]{./images/experimental_setup.png}
\small\color{gray}Two-zone environment: door separates A from B

::::
:::

::: notes

- Let me formalize the problem. We have a bus door, and an access point nearby.
- Zone A is inside the bus. Zone B is the outside, the bus stop area.
- Any person near the door is in one of four states: inside and staying inside (AA), outside and staying outside (BB), moving from outside to inside, boarding (BA), or moving from inside to outside, alighting (AB).
- This is the fundamental classification we need to solve.
- The notation uses arrows: B→A means boarding, A→B means alighting, and the static classes are AA and BB.

:::

# The Idea

## Using Wi-Fi Signals as Movement Signatures

\centering

\vspace{1em}

**One access point. Ten seconds of RSSI. Four movement classes.**

\vspace{2em}

::: columns
:::: {.column width="30%"}

\centering
\textbf{No specialized hardware}

Standard Wi-Fi AP already on many buses

::::
:::: {.column width="30%"}

\centering
\textbf{No device identification}

We track signal patterns, not people

::::
:::: {.column width="30%"}

\centering
\textbf{No active participation}

Uses existing Wi-Fi associations passively

::::
:::

\vspace{2em}

\small\color{gray}The temporal evolution of RSSI encodes whether a device is moving toward or away from the access point.

::: notes

- The core idea is simple. If you hold a Wi-Fi device and walk toward an access point, the signal gets stronger. Walk away, it gets weaker. Stay still, it stays flat.
- We capture this over a 10-second window at 1 sample per second, 10 RSSI values.
- That is all we need. One AP at the door. No cameras. No pressure mats. No special hardware.
- And critically: we do not identify devices. We just need to know that a signal exists and how it changes. The privacy implication is fundamental.
- The question is whether machine learning can learn these temporal patterns reliably.

:::

# How We Collected the Data

## Experimental Setup

::: columns
:::: {.column width="40%"}

\centering
\includegraphics[width=\linewidth]{./images/experimental_setup.png}
\small\color{gray}MAYBE CHANGE HERE TO A REAL PHOTO?? Controlled indoor environment at IT Aveiro

::::
:::: {.column width="60%"}

**Physical setup:**

- Zone A: closed room (bus interior), AP at doorway
- Zone B: corridor (bus stop), wall attenuation between zones

**Collection:**

- 10 s windows at 1 Hz $\rightarrow$ 10 RSSI values per sample
- 4 smartphones, 3 brands
- Scripted movements with ground truth labels
- MAC-based device isolation for clean trajectories

::::
:::

::: notes

- We built a controlled environment at our institute. A closed room represents the bus interior, and the corridor outside represents the bus stop.
- The access point sits right at the doorway, exactly where you would mount it on a real bus.
- We used four different smartphone models: Samsung Galaxy S20, Samsung Galaxy S23, POCO X7 Pro, and Xiaomi Redmi 4. Three brands, spanning Android 6 to Android 14, different Wi-Fi chipsets. This diversity is essential for generalization.
- Each trial is exactly 10 seconds long, with RSSI sampled once per second, so 10 values per observation window.
- In the noisy scenario, four devices operated simultaneously in paired movements — some boarding while others stayed static, creating realistic co-channel interference.
- Because we controlled the environment, we knew the ground truth for every trial, each 10-second window was labeled as AA, BB, AB, or BA.

:::

# The Dataset

## What We Collected

**1,356 labelled samples** across two conditions:


| Condition    | Samples | Description                             |
| ------------ | ------- | --------------------------------------- |
| **Isolated** | 160     | One device at a time, clean signals     |
| **Noisy**    | 1,196   | Four simultaneous devices, interference |
\small Approximately 340 samples per movement class. The dataset is publicly available on IEEE Dataport and Zenodo.

\centering
\scriptsize
\rowcolors{2}{lightgray!15}{white}
\begin{tabular}{c|cccccccccc|c}
\toprule
\textbf{ID} & \textbf{t1} & \textbf{t2} & \textbf{t3} & \textbf{t4} & \textbf{t5} & \textbf{t6} & \textbf{t7} & \textbf{t8} & \textbf{t9} & \textbf{t10} & \textbf{Class} \\
\midrule
1  & −42 & −44 & −48 & −52 & −55 & −58 & −60 & −62 & −63 & −64 & AB \\
2  & −65 & −63 & −58 & −54 & −50 & −47 & −45 & −43 & −42 & −41 & BA \\
3  & −40 & −41 & −39 & −42 & −40 & −41 & −39 & −40 & −41 & −40 & AA \\
4  & −62 & −63 & −61 & −64 & −62 & −63 & −61 & −62 & −63 & −62 & BB \\
\bottomrule
\end{tabular}
\small\color{gray}Four samples from the dataset: each row is a 10-second RSSI trajectory

::: notes

- Our dataset has 1,356 samples, each one a 10-second window with 10 RSSI values and a class label.
- We collected under two conditions: isolated (single device, clean signals) and noisy (four devices transmitting simultaneously).
- The noisy scenario is much closer to reality: in a real bus stop you have multiple people, some boarding, some waiting, some just passing by. These additional devices create overlapping RSSI patterns, and the classifier must cut through that noise.
- The isolated subset is small, only 160 samples, but it gives us an upper bound on what is possible without interference.
- The dataset is publicly available on IEEE Dataport and Zenodo. We want other researchers to build on this work.
- Look at this sample: rows 1 and 2 show alighting (decreasing RSSI) and boarding (increasing RSSI). These temporal patterns, not the absolute values, are what the classifier learns.

:::

# What the Signals Look Like

## Temporal RSSI Trajectories

::: columns
:::: {.column width="55%"}

\centering
\includegraphics[width=\linewidth]{./images/rssi_mean_trajectory_per_class.pdf}
\small\color{gray}Mean RSSI trajectory per class over 10 seconds (± 1 std)

::::
:::: {.column width="45%"}

**Clear temporal signatures emerge:**

- **Boarding (BA):** steady RSSI increase as the device approaches the AP
- **Alighting (AB):** pronounced RSSI decrease as the device moves away
- **Static states (AA, BB):** stable signal at distinct levels: AA higher (near AP), BB lower (outside)

\vspace{0.4em}
These contrasting trends are the foundation of the classification.

::::
:::

::: notes

- Here we see why the problem is solvable. Each movement class has a distinctive RSSI signature over time.
- Boarding, the green line, shows signal strength climbing as the person walks toward the access point.
- Alighting, the red line, shows signal dropping as they walk away through the door.
- The two static classes sit at different levels: AA is higher because the device is inside near the AP; BB is lower because it is outside, behind a wall.
- There is overlap in standard deviations, this is not a trivial problem. But the mean trends are distinct enough that machine learning can separate them.
- Remember: each of these lines is just 10 RSSI values. That is our entire feature space.

:::

# Are the Classes Separable?

## t-SNE Visualization

::: columns
:::: {.column width="55%"}

\centering
\includegraphics[width=\linewidth]{./images/tsne_visualization.pdf}
\small\color{gray}t-SNE projection of the 10-dimensional RSSI feature space

::::
:::: {.column width="45%"}

\vspace{3.5em}
**Classes form well-defined clusters** confirming that the collected samples carry structure for classification.

\vspace{0.5em}
**Overlap between AA and BA expected**: both involve strong RSSI near the AP, but differ in temporal pattern.

::::
:::

::: notes

- t-SNE projects our 10-dimensional data into 2D so we can see the structure.
- The four classes form distinct clusters. They are not perfectly separated, but they are clearly structured.
- The overlap between AA and BA makes physical sense: both involve a device near the access point with high RSSI. The difference is whether the signal is stable (AA) or increasing (BA).
- This overlap is what makes the problem interesting, and where machine learning adds value over simple thresholding.
- The important message: yes, 10 RSSI values are enough. The data has structure.

:::

# Our Machine Learning Framework

## How We Evaluated the Classifiers

::: columns
:::: {.column width="48%"}

**38 classifiers** from six families:

- SVMs (RBF, Linear)
- Ensemble methods (Random Forest, Extra Trees, CatBoost, XGBoost, LightGBM)
- Gaussian Process
- Neural Networks (MLPs, small to large)
- Regularized logistic regression (L1, L2, ElasticNet)
- Stacking and Voting ensembles

::::
:::: {.column width="48%"}

**Rigorous evaluation protocol:**

- Stratified 80/20 train-test split
- 5-fold stratified cross-validation
- 3 random seeds for stability
- Bayesian hyperparameter optimization (Optuna, 50–1300+ trials per classifier)

\vspace{0.5em}

**Primary metric: Matthews Correlation Coefficient (MCC)**

\small It produces high scores only when all four quadrants of the confusion matrix perform well.

::::
:::

::: notes

- We did not just try one or two classifiers. We evaluated 38 different algorithms across six families.
- Every classifier underwent Bayesian hyperparameter optimization with Optuna, with up to 1,300 trials per model. This ensures a fair comparison.
- We used three random seeds and report mean and standard deviation. Reproducibility matters.
- Now, the most important metric. We chose MCC, Matthews Correlation Coefficient, as our primary metric. Why? Because accuracy alone is misleading for multi-class problems. MCC only gives high scores when the model performs well across all four confusion matrix quadrants, true positives, true negatives, false positives, and false negatives for every class.
- MCC is the standard in biomedical machine learning and increasingly in other fields. It is a single number you can trust.

:::

# Results: Performance Across Scenarios

## MCC Comparison

::: columns
:::: {.column width="50%"}

\centering
\includegraphics[width=\linewidth]{./images/mcc_variability.pdf}
\small\color{gray}MCC variability across three random seeds (combined dataset)

::::
:::: {.column width="50%"}

| Classifier       | Combined  | Isolated  | Noisy     |
| ---------------- | --------- | --------- | --------- |
| Gaussian Process | **0.756** | 0.414     | 0.755     |
| SVC (RBF)        | 0.755     | 0.825     | 0.754     |
| CatBoost         | 0.746     | 0.782     | **0.770** |
| KNN (k=5)        | 0.692     | **0.907** | 0.704     |
\small Mean MCC ± std across 3 seeds

\vspace{0.5em}

::::
:::

::: notes

- This table shows our headline results across the three scenarios.
- Gaussian Process achieved the best MCC on the combined dataset at 0.756.
- Look at the isolated scenario: KNN hits 0.907 MCC, near-perfect classification when there is no interference. This tells us the signal patterns themselves are highly discriminative.
- But Gaussian Process on isolated drops to 0.414. Why? The isolated set has only 160 samples, too few for the GP kernel to estimate reliably. This is a sample size limitation, not a model failure.
- The noisy scenario, closest to reality, shows CatBoost at 0.770. That is our most realistic performance estimate.
- The noisy results are consistent with the combined results because noisy data makes up 88% of the dataset.

:::

# Results: Confusion Patterns

## Gaussian Process on Combined Dataset

::: columns
:::: {.column width="54%"}

\centering
\includegraphics[width=\linewidth]{./images/confusion_matrix_GaussianProcess.pdf}
\small\color{gray}Normalized confusion matrix, Gaussian Process

::::
:::: {.column width="42%"}

\small
| Class | Precision | Recall | F1 |
| ------------------ | --------- | ------ | ----- |
| AA (Inside) | 0.78 | 0.73 | 0.75 |
| BB (Stop) | 0.87 | 0.89 | 0.88 |
| BA (Boarding) | 0.83 | 0.77 | 0.80 |
| AB (Alighting) | 0.78 | 0.88 | 0.83 |
| **Weighted Avg** | **0.82** | **0.82**|**0.82**|

\vspace{0.5em}
\small Strong diagonal dominance. Errors are concentrated between spatially adjacent classes: AA/BA and AB/BB.

::::
:::

::: notes

- The confusion matrix shows where we succeed and where we struggle.
- Diagonal dominance is strong. Most predictions are correct.
- The main confusions are between AA and BA, both near the access point, both with strong RSSI, and between AB and BB, both with weaker signals.
- Notice that BB (waiting at the stop) is the easiest class: 0.88 F1. The physical barrier between zones creates a clear signal separation.
- AA (inside the bus) is the hardest: 0.75 F1. When a device is already near the AP, the temporal signal of boarding versus staying is subtle.
- This tells us where future work should focus: distinguishing static from transitional states when the RSSI magnitude is similar.

:::

# Results: Clean vs. Noisy Data

## The Effect of Interference

\centering

| Scenario     | Best Classifier | MCC   | AA F1 | BB F1 | BA F1 | AB F1 |
| ------------ | --------------- | ----- | ----- | ----- | ----- | ----- |
| **Isolated** | KNN (k=5)       | 0.907 | 0.87  | 0.96  | 0.91  | 0.96  |
| **Noisy**    | CatBoost        | 0.770 | 0.79  | 0.87  | 0.82  | 0.83  |
| **Combined** | Gaussian Proc.  | 0.756 | 0.75  | 0.88  | 0.80  | 0.83  |

\vspace{0.8em}

::: columns
:::: {.column width="45%"}

\centering
\small\textbf{Clean signals: near-perfect}

F1 scores above 0.86 for every class. Simple classifiers (KNN) perform best. Upper bound on what RSSI can achieve.

::::
:::: {.column width="45%"}

\centering
\small\textbf{Realistic conditions: robust}

Per-class F1 above 0.77 with interference. Ensemble methods handle noise best.

::::
:::

::: notes

- This slide makes the clean-versus-noisy comparison explicit.
- Under clean, isolated conditions, we get near-perfect classification. KNN with k=5 reaches 0.907 MCC. BB and AB have F1 scores of 0.96. This shows the promise of the approach.
- Under noisy, realistic conditions, performance drops but remains solid. CatBoost achieves 0.770 MCC, with per-class F1 above 0.77. Ensemble methods handle interference better than simpler classifiers.
- The gap between 0.907 and 0.770 is the cost of interference. Closing this gap, making the system robust to noise, is our main focus going forward.
- But even 0.770 MCC is operationally useful as a complementary technology. It does not need to replace APC systems; it augments them.
- Gradient boosting methods like CatBoost and XGBoost showed particular robustness to signal interference, making them our recommended choice for deployment.

:::

# Feature Importance

## What the Classifiers Actually Use

::: columns
:::: {.column width="50%"}

\centering
\includegraphics[width=\linewidth]{./images/mean_feature_importance.pdf}
\small\color{gray}Mean feature importance across interpretable classifiers (min-max normalized)

::::
:::: {.column width="50%"}

\setlength{\leftmargini}{0.8em}
\begin{itemize}
\item \textbf{Feature 1}: initial RSSI, universal classifier agreement
\item \textbf{Feature 6}: mid-trajectory, confirms movement direction
\item \textbf{Feature 2}: early trajectory, reinforces starting position
\item \textbf{Feature 10}: end of window, confirms movement direction
\end{itemize}

\small Sequential measurements outperform aggregate statistics. The temporal order matters.

::::
:::

::: notes

- Feature importance reveals that the first RSSI measurement is overwhelmingly the most important, all classifiers agree on this (standard deviation 0.009).
- The first three samples essentially tell the classifier where the device starts. Feature 6, at the middle of the window, captures whether the device is moving and in which direction.
- The later features (7–10) matter less because by mid-window, the trajectory is largely determined.
- This has a practical implication: we might be able to shorten the observation window and still maintain performance. That matters for faster boarding scenarios.
- The key insight: temporal order matters. Sequential RSSI values carry information that aggregate statistics like mean and variance lose.

:::

# Discussion

## What These Results Mean

::: columns
:::: {.column width="48%"}

**What we have demonstrated:**

- **Four classes separable** from 10 RSSI values
- **MCC 0.907** in clean conditions, **0.770** with interference
- **Single access point** at the door, no extra hardware
- **Privacy-preserving**: no device identification, no localization

::::
:::: {.column width="48%"}

**What we must be careful about:**

- **Controlled environment**: real buses add vibration, crowding, body absorption
- **10 s window** may not fit all movement speeds
- **Only associated devices** are visible; Wi-Fi-off passengers go undetected

::::
:::

::: notes

- Let me be clear about what we have and have not shown.
- We have demonstrated that the approach works in a controlled environment. MCC of 0.907 in clean conditions proves the signal patterns are discriminative. MCC of 0.770 in noisy conditions shows the approach is robust enough to be useful.
- But I want to be transparent about limitations. This is a lab experiment. On a real bus, you have vehicle vibration, metal surfaces reflecting signals, crowds of people absorbing RF, and passengers moving at different speeds. All of these degrade RSSI separability.
- Also, our system only sees devices that are associated with the access point. Passengers with Wi-Fi off, using mobile data exclusively, or whose devices do not auto-connect are invisible. Modern MAC address randomization during probe requests does not affect us since we rely on Wi-Fi association, but first-time passengers who never connect will not be counted.
- The isolated dataset is small. We report those numbers because they are informative, but do not overinterpret them.
- These limitations are not blockers. They define our research agenda.

:::

# Conclusions and Future Work

## Where We Go from Here

::: columns
:::: {.column width="45%"}

**This work:**

- **Temporal RSSI classification** of 4 movement patterns
- **38 classifiers** evaluated with Bayesian optimization
- **MCC 0.756** combined, **0.907** clean, **0.770** noisy

\vspace{0.5em}

**The dataset is public.**

\footnotesize IEEE Dataport · Zenodo · GitHub

::::
:::: {.column width="45%"}

**Next steps:**

- **Operational validation**: real bus deployment
- **Adaptive windows**: dynamic length based on movement speed
- **Sensor fusion**: combine with other modalities
- **OD matrix estimation**: aggregate events across stops
- **Noise robustness**: close the 0.907 $\rightarrow$ 0.770 gap

::::
:::

::: notes

- To wrap up: we proposed a new way to classify passenger movements using only Wi-Fi RSSI. No cameras, no pressure mats, no specialized hardware.
- The results are promising, not definitive. We showed that the approach works in controlled conditions. The next step is to take it to a real bus.
- We are particularly excited about adaptive observation windows: can the system decide how long to observe based on what it sees?
- Sensor fusion is another key direction: combining RSSI with other low-cost modalities could make the system robust enough for production deployment.
- OD matrix estimation is the ultimate goal: aggregate individual boarding and alighting events across stops to recover full origin-destination flows for the network.
- The dataset is public. We encourage the community to build on this work, try new classifiers, new features, new deployment scenarios.
- Finally, we want to close the gap between clean (0.907) and noisy (0.770). That 0.137 MCC difference is where the real research lives.

:::

# Acknowledgements

\centering

\vspace{2em}

**This work is supported by the European Regional Development Fund (FEDER)**

through the Regional Operational Programme of Centre (CENTRO 2030)

of the Portugal 2030 framework

\vspace{1em}

Project **inMotion**, Nr. 21359 (CENTRO2030-FEDER-02225900)

\vspace{2em}

\small\textbf{Instituto de Telecomunicações · Universidade de Aveiro}

\vspace{1em}

\normalsize Thank you. Questions?

\vspace{1em}

\footnotesize Dataset: IEEE Dataport (10.21227/55nm-0r91) · Code: github.com/ATNoG/inMotion

::: notes

- Before I take questions, I want to acknowledge the funding that made this possible. This work is part of the inMotion project, funded by European structural funds through CENTRO 2030.
- The dataset and code are publicly available, the links are on the slide.
- I am happy to take your questions now.

:::
