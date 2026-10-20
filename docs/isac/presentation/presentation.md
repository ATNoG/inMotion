---
title: Passive ISAC for Urban Transit using Self-Supervised World Models
subtitle: Opportunistic Wi-Fi Sensing for Passenger Monitoring
author: "\\underline{André Ribeiro}, Guilherme Matos, Julio Corona, Mário Antunes, Diogo Gomes"
aspectratio: 169
institute: Instituto de Telecomunicações, Universidade de Aveiro, Portugal
date: 26/10/2026 - FNWF26, Lisbon
section-titles: false
# toc: true
# toc-title: "Table of Contents"
header-includes:
  - \usetheme[sectionpage=none,numbering=fraction,progressbar=frametitle]{moloch}
  - \setbeamertemplate{navigation symbols}{}
  - \usepackage[semibold,light]{FiraSans}
  - \usepackage{FiraMono}
  - \usepackage{longtable,booktabs}
  - \usepackage[table]{xcolor}
  - \usepackage{etoolbox}
  - \AtBeginSection{}
  - \AtBeginSubsection{}
---

## The inMotion Project

::: columns
:::: {.column width="52%"}

The **inMotion** project aims to bring real-time intelligence to public transport operations.

One of its core goals, understand **how passengers move** through the network: who boards, who alights, and where.

A previous paper answered that with hand-crafted RSSI features and supervised classifiers. Today the question is different: can a **world model learn the dynamics** itself?

::::
:::: {.column width="44%"}

\centering
\includegraphics[width=0.85\linewidth]{./images/smart_bus.png}
\small\color{gray}Smart bus concept

::::
:::

::: notes

- Good morning. This work is part of the inMotion project, supported by CENTRO 2030.
- The project's mission is to give public transport operators real-time knowledge of how passengers move: who boards, who alights, and where.
- At EuCNC we showed that this is possible with hand-crafted RSSI features and supervised classifiers. The ceiling was a Matthews Correlation Coefficient of 0.756.
- This talk is about what changed: instead of engineering the features, we let a self-supervised world model learn the radio dynamics, and the metric goes from 0.756 to 0.9194.
- Everything you see is one access point already on the bus. No camera, no radar, no new hardware.

:::

## The Passenger Counting Gap

::: columns
:::: {.column width="48%"}

**Without granular passenger data:**

- Routes designed on rough estimates, not actual demand
- Buses run near-empty on some segments, overcrowded on others
- Operators cannot justify service adjustments to funders
- Fleet sizing based on guesswork, not occupancy

::::
:::: {.column width="48%"}

**The potential of data-driven optimization:**

| Metric           | Improvement              |
| ---------------- | ------------------------ |
| Route costs      | $\downarrow$ **30–40\%** |
| Fuel consumption | $\downarrow$ **10–20\%** |
| Routes needed    | $\downarrow$ **29–42\%** |

\vspace{0.3em}
\footnotesize\color{gray}Sources: UPS ORION (30M liters fuel/year saved); industry route optimization benchmarks

::::
:::

::: notes

- Accurate passenger counting is the input to every operational decision an operator makes.
- Traditional systems, infrared beams, pressure mats, cameras, work reasonably in the lab. In the field, with crowds, occlusion and dirt, claimed 98 percent becomes 53 percent. Pronello and Ruiz documented that gap.
- But the financial case is what convinces operators. UPS saved 30 million liters of fuel per year by optimizing routes with ORION. Industry route planning cuts delivery costs 30 to 40 percent.
- The same logic applies to public transport. If you know exactly where people board and alight, you can right-size vehicles and cut empty runs. The numbers on the right are the prize.
- Wi-Fi is already installed on many buses. The infrastructure exists. This work is about extracting the signal from it.

:::

## Four Movements to Classify

\centering
\includegraphics[width=0.96\linewidth]{./images/fig_setup.pdf}

\vspace{0.1em}
\begin{minipage}{0.94\linewidth}
\footnotesize
\textbf{Four states:} \textbf{AA} cabin transit · \textbf{AB} alighting · \textbf{BA} boarding · \textbf{BB} platform waiting.

\textbf{One observation:} 10 s at 1 Hz, so 10 RSSI values per window.
\end{minipage}

::: notes

- Formalise the problem. There is a door, and one access point sits right at it.
- Zone A is the cabin: an enclosed room, strong signal, roughly minus 40 dBm. Zone B is the platform: a corridor behind a wall, roughly minus 52 dBm.
- Every person near the door is in one of four states. Inside and staying inside is AA, cabin transit. Platform and staying is BB, platform waiting. Moving from the platform into the cabin is BA, boarding. Moving out is AB, alighting.
- The physics is in the level, but the semantics is in the _trend_. That is the whole difficulty.
- One observation window is ten seconds sampled at one hertz, so ten RSSI values. That is the raw input to everything I show later.

:::

## Wi-Fi RSSI Is a Movement Signature

::: columns
:::: {.column width="54%"}

\centering
\includegraphics[width=\linewidth]{./images/rssi_trajectory.pdf}
\small\color{gray}Mean RSSI per class on the 3,511-sample collection ($\pm 1\sigma$)

::::
:::: {.column width="44%"}

**The trend carries the class:**

- **Boarding (BA):** signal rises from $-51$ to $-43$ dBm as the device approaches the AP
- **Alighting (AB):** falls from $-42$ to $-52$ dBm as it walks away
- **Cabin transit (AA):** stable at $\approx -40$ dBm
- **Platform waiting (BB):** stable at $\approx -52$ dBm

\vspace{0.3em}
The $\pm 1\sigma$ bands overlap heavily. Classes differ in **shape**, not in level.

::::
:::

::: notes

- This is why the problem is solvable at all, and also why it is hard.
- Boarding, the purple line, climbs about eight dBm as the person walks toward the access point. Alighting, the orange line, falls about ten dBm as they walk away.
- The two static classes sit at different levels: AA is high because the device is inside near the AP, BB is low because it is outside behind a wall.
- Now look at the shaded bands. The standard deviations overlap heavily. If you only looked at the level, you could not separate boarding from standing inside, because both are strong.
- What separates them is the temporal shape. That is exactly what a world model is good at learning, and exactly what a hand-built threshold is bad at.

:::

## How a World Model Learns

\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[x=1cm,y=1cm,>=stealth]
\definecolor{ink}{HTML}{37474F}
\definecolor{wmpurp}{HTML}{7570B3}
\definecolor{wmorg}{HTML}{D95F02}
\definecolor{wmslate}{HTML}{7F8C8D}
\tikzset{
pan/.style={draw=wmslate!45, fill=wmslate!4, rounded corners=6pt, line width=0.5pt},
badge/.style={circle, draw=none, fill=ink, inner sep=0pt, minimum size=0.30cm, font=\tiny\bfseries, text=white},
ptitle/.style={anchor=west, font=\scriptsize\bfseries, text=ink},
pcap/.style={text width=3.3cm, align=center, font=\tiny, text=ink},
trace/.style={draw=ink, line width=0.9pt},
traced/.style={draw=ink, line width=0.9pt, dashed},
base/.style={draw=wmslate!45, line width=0.4pt}
}

% ---------- 1. WATCH ----------
\begin{scope}[shift={(0,0)}]
\draw[pan] (0,0) rectangle (4.0,-3.4);
\node[badge] at (0.32,-0.34) {1};
\node[ptitle] at (0.55,-0.34) {Watch};
\draw[base] (0.30,-2.30) -- (3.70,-2.30);
\draw[trace] plot coordinates {(0.35,-1.95) (0.72,-1.85) (1.09,-1.90) (1.46,-1.70) (1.83,-1.66) (2.20,-1.48) (2.57,-1.42) (2.94,-1.33) (3.31,-1.30) (3.68,-1.22)};
\node[pcap] at (2.0,-2.92) {Ten seconds of radio from one access point. No camera, no label.};
\end{scope}

% ---------- 2. GUESS ----------
\begin{scope}[shift={(4.3,0)}]
\draw[pan] (0,0) rectangle (4.0,-3.4);
\node[badge] at (0.32,-0.34) {2};
\node[ptitle] at (0.55,-0.34) {Guess};
\draw[base] (0.30,-2.30) -- (3.70,-2.30);
\draw[trace] plot coordinates {(0.35,-1.95) (0.72,-1.85) (1.09,-1.90) (1.46,-1.70) (1.83,-1.66)};
\draw[traced] plot coordinates {(1.83,-1.66) (2.20,-1.48) (2.57,-1.42) (2.94,-1.33) (3.31,-1.30) (3.68,-1.22)};
\node[circle, draw=ink!45, fill=white, inner sep=1pt, minimum size=0.30cm, font=\tiny\bfseries, text=ink] at (2.75,-1.05) {?};
\node[pcap] at (2.0,-2.92) {Hide the second half. Guess it --- as an idea, not as a waveform.};
\end{scope}

% ---------- 3. UNDERSTAND ----------
\begin{scope}[shift={(8.6,0)}]
\draw[pan] (0,0) rectangle (4.0,-3.4);
\node[badge] at (0.32,-0.34) {3};
\node[ptitle] at (0.55,-0.34) {Understand};
\draw[ink, line width=0.9pt] (1.95,-1.10) -- (1.95,-2.30) -- (2.45,-2.30) -- (2.45,-1.10);
\draw[->, draw=wmpurp, line width=1.1pt] (0.45,-1.50) -- (1.72,-1.50);
\node[font=\tiny, text=wmpurp] at (1.08,-1.36) {BA};
\draw[->, draw=wmorg, line width=1.1pt] (2.68,-1.50) -- (3.78,-1.50);
\node[font=\tiny, text=wmorg] at (3.23,-1.36) {AB};
\draw[wmpurp, line width=0.8pt] plot coordinates {(0.55,-2.10) (0.95,-2.00) (1.35,-1.90)};
\draw[wmorg, line width=0.8pt] plot coordinates {(2.95,-1.90) (3.35,-2.00) (3.72,-2.10)};
\node[pcap] at (2.0,-2.80) {What stays is motion:\\ toward the door, or away.};
\end{scope}

\node[text width=12.4cm, align=center, font=\small, text=ink] at (6.3,-4.06)
{\textbf{It never sees the answer.} That alone beats every supervised model here --- and it keeps getting better as we add data.};
\node[text width=12.6cm, align=center, font=\footnotesize, text=wmslate] at (6.3,-4.82)
{The world here is the radio field around the door; the physics is a passenger walking.};
\end{tikzpicture}%
}

::: notes

- This is the slide to remember, and there is no maths on it.
- Forget networks for a second. Think of a child learning what a ball does. Nobody hands it a labelled dataset. It watches, guesses what happens next, and is surprised when it is wrong. That is all a world model is, formalised.
- Here the child sees ten seconds of radio. We hide the second half and make it guess. It is scored only on that guess.
- The non-obvious part: it guesses in its own compressed internal language, not in the signal. We do not ask it to redraw the waveform. Redrawing the waveform means redrawing the noise, and the noise is exactly what differs between one bus, one phone and one season.
- The crazy part: nobody ever tells it that this window is a boarding. It never sees the word. And that is enough: it beats every supervised model we trained on this task, and it gets better, not worse, as we add data.
- Now the transposition. In a language model the world is text. Here the world is the radio field around one door, and the physics inside it is a person walking. The signal rises as they approach, falls as they leave.
- That is why it transfers: motion is invariant, multipath is not.
- If you want the machinery, the next slide has it.

:::

## Predicting in Latent Space

\centering
\resizebox{\linewidth}{!}{%
\begin{tikzpicture}[x=1cm,y=1cm,>=stealth]
\definecolor{wmteal}{HTML}{1B9E77}
\definecolor{wmpurp}{HTML}{7570B3}
\definecolor{wmorg}{HTML}{D95F02}
\definecolor{wmamb}{HTML}{E6AB02}
\definecolor{wmslate}{HTML}{7F8C8D}
\tikzset{
enc/.style={draw=wmteal, fill=wmteal!8, rounded corners=2pt, line width=0.6pt, align=center, inner sep=2.5pt, font=\scriptsize},
prd/.style={draw=wmpurp, fill=wmpurp!8, rounded corners=2pt, line width=0.6pt, align=center, inner sep=2.5pt, font=\scriptsize},
tgt/.style={draw=wmorg, fill=wmorg!8, rounded corners=2pt, line width=0.6pt, align=center, inner sep=2.5pt, font=\scriptsize},
los/.style={draw=wmamb, fill=wmamb!12, rounded corners=2pt, line width=0.8pt, align=center, inner sep=2.5pt, font=\scriptsize},
dat/.style={draw=wmslate, fill=wmslate!8, rounded corners=2pt, line width=0.6pt, align=center, inner sep=2.5pt, font=\scriptsize},
ar/.style={draw=wmslate, line width=0.6pt, ->},
ard/.style={draw=wmslate, line width=0.6pt, ->, dashed}
}
\node[dat, text width=2.1cm, minimum height=0.75cm] (xt) at (1.25,0) {RSSI window\\$x_t$ \tiny(10 s)};
\node[enc, text width=2.6cm, minimum height=0.75cm] (enc) at (4.75,0) {\textbf{Context encoder}\\$f_\theta$};
\node[prd, text width=2.6cm, minimum height=0.75cm] (prd) at (8.65,0) {\textbf{Latent predictor}\\$h_\psi$};
\node[dat, text width=1.9cm, minimum height=0.75cm] (hat) at (11.90,0) {$\hat{s}_{t+k}$\\\tiny predicted latent};
\node[dat, text width=2.6cm, minimum height=0.75cm] (crop) at (4.75,-2.50) {masked / future\\crop};
\node[tgt, text width=2.6cm, minimum height=0.75cm] (tenc) at (8.65,-2.50) {\textbf{Target encoder}\\$g_\xi$ \scriptsize(EMA)};
\node[los, text width=2.1cm, minimum height=0.80cm] (loss) at (11.90,-1.25) {$\|\hat{s}-\mathrm{sg}(z)\|_2^2$\\\tiny L2, in latent space};
\draw[ar] (xt.east) -- (enc.west);
\draw[ar] (enc.east) -- node[above, font=\scriptsize] {$s_t$} (prd.west);
\draw[ar] (prd.east) -- (hat.west);
\draw[ard] (xt.south) |- (crop.west);
\draw[ar] (crop.east) -- (tenc.west);
\draw[ar] (tenc.east) -| (loss.south);
\draw[ar] (hat.south) -- (loss.north);
\draw[dashed, draw=wmslate, line width=0.5pt, rounded corners=5pt] (-0.15,0.44) rectangle (13.05,-2.92);
\node[anchor=west, font=\tiny, text=wmslate] at (-0.13,0.70) {self-supervised objective \textperiodcentered{} no labels \textperiodcentered{} no output reconstruction};
\end{tikzpicture}%
}

\vspace{0.1em}
\begin{minipage}{0.94\linewidth}
\footnotesize
\textbf{No decode back to the signal.} The target encoder is a slow EMA copy of the context encoder and its output is stop-gradiented, so the two branches cannot collapse onto each other; the loss is an L2 distance between two \emph{latent} vectors.
\end{minipage}

::: notes

- Now the machinery, for the record.
- The left branch encodes a window into a latent vector. The predictor is asked for the latent of a masked or future piece of the same window.
- The bottom branch is the same encoder, kept as a slowly updated EMA copy, and its output goes through a stop-gradient. That asymmetry is what stops the two branches from collapsing onto the same trivial answer.
- The loss on the right is an L2 distance between two latent vectors. No label, no reconstruction, no decode.
- Why this matters: the prediction problem forces the representation to keep what physically evolves and to drop what does not.
- The four variants in the paper differ only in how they tokenise time and where they put the mask. The idea is identical.

:::

## Data Collection Setup

::: columns
:::: {.column width="40%"}

\centering
\includegraphics[width=0.82\linewidth]{./images/real_env.jpg}

\small\color{gray}Controlled indoor environment, IT Aveiro

::::
:::: {.column width="58%"}

**Physical setup:**

- Zone A: closed room (cabin), AP at the doorway
- Zone B: corridor (platform), wall attenuation between zones

**Collection:**

- 10 s windows at 1 Hz $\rightarrow$ 10 RSSI values
- **7 smartphone models**, multiple vendors, multiple Android releases
- Scripted movements with ground-truth session labels
- Passive sniffing: no association, no identification

::::
:::

::: notes

- We built a controlled environment at the institute. A closed room represents the cabin, and the corridor outside represents the platform.
- The access point sits exactly at the doorway, where it would be mounted on a vehicle.
- We collected from seven distinct smartphone models across multiple vendors and several Android releases. This matters: the radio chipset changes the RSSI curve, and a model that has only seen one phone will not generalise.
- Each trial is ten seconds, sampled once per second.
- Movements were scripted and the label comes from the session protocol, not from a second annotator. Keep that in mind when we discuss limitations.
- The sniffer is passive. We never associate with the phone, we never ask it to identify itself.

:::

## Dataset Composition and Splits

\centering
\footnotesize
\setlength{\tabcolsep}{8pt}
\begin{tabular}{lrl}
\toprule
\textbf{Condition} & \textbf{Samples} & \textbf{Description} \\
\midrule
Single-device & 160 & pure signals, 40 per class \\
Multi-device & 3,351 & concurrent interference, $\approx 840$ per class \\
\textbf{Total} & \textbf{3,511} & primary benchmark \\
\bottomrule
\end{tabular}

\vspace{0.7em}
\begin{minipage}{0.47\linewidth}
\footnotesize
\textbf{Strict superset.} The earlier 2,251-sample collection is fully contained: all eight earlier devices are retained, and five more are added.
\end{minipage}\hfill
\begin{minipage}{0.49\linewidth}
\footnotesize
\textbf{Protocol.} 80/20 class-stratified split, with 12.5\% of training held out for validation. Gaussian noise ($\sigma=0.05$) and intra-class Mixup ($\alpha=0.4$) give \textbf{35,839} training windows; the test split is evaluated once.
\end{minipage}

::: notes

- The primary dataset is 3,511 labelled windows. 160 are single-device calibration traces with clean signals, 40 per class. The other 3,351 are multi-device traces, roughly 840 per class, where up to four phones transmit at the same time.
- The noisy part is the realistic part. On a real platform you have people boarding, waiting and walking past, all with active phones.
- This collection is a strict superset of the 2,251-sample campaign from our earlier paper: every device is retained and five more are added. That is deliberate. It lets us measure what happens to each model family when the collection grows.
- Augmentation happens inside the training folds only, with Gaussian noise and Mixup. That gives about 36,000 training windows with no contamination of the split.
- The test split is evaluated once, with the model selected on validation MCC.

:::

## The Four World Models

\centering
\scriptsize
\setlength{\tabcolsep}{7pt}
\begin{tabular}{lcccc}
\toprule
& \textbf{TS-JEPA} & \textbf{LeJEPA} & \textbf{CF-JEPA} & \textbf{T-JEPA} \\
\midrule
Input channels & 18 (rich) & 4 (base) & 18 (rich) & 18 (rich) \\
Tokenizer & 1D conv patch & causal temporal & 1D conv patch & per-feature + REG \\
Patch / token size & 2 timesteps & 1 timestep & 2 timesteps & 1 timestep \\
Embedding dim / heads & 256 / 8 & 128 / 4 & 256 / 8 & 128 / 4 \\
Encoder layers & 4 & 4 & 3 & 4 \\
Predictive objective & masked patch L2 & next-step L2 & multi-horizon L2 & masked token L2 \\
Anti-collapse & EMA target & SIGReg & EMA target & EMA + REG \\
Parameters & 2.153 M & \textbf{0.566 M} & 1.232 M & 0.633 M \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\begin{minipage}{0.95\linewidth}
\footnotesize
\textbf{Baselines:} Mamba-3 selective state spaces (CNN, TCN, Transformer, multi-view; 0.84–1.70 M) · SIGReg regularized CNN (0.697 M) · DeepStack two-stage stack (43.6 M) · Optuna-tuned GRU.

\textbf{Interpretability:} SHAP attributions over the 18 channels, plus architecture-controlled ablation.
\end{minipage}

::: notes

- All four models share one objective: predict the representation of a held-out region of the window, in latent space, without labels.
- TS-JEPA is our strongest variant. It uses 18 input channels, a 1D convolutional patch tokenizer with patch size two, and patch masking annealed from 0.4 to 0.7 during pretraining.
- LeJEPA is the compact one. Four base channels, causal next-step prediction, no EMA target: it is stabilised instead by a sketched isotropic Gaussian regulariser. It has 0.566 million parameters, which is why it is the edge candidate.
- CF-JEPA predicts across three horizons without any masking, and T-JEPA tokenises each feature separately with a learned register token.
- The baselines matter for the story: the strongest supervised and state-space models, tuned with Optuna and given equal compute, are what we compare against.
- Pretraining is 500 epochs with AdamW and cosine annealing, then fine-tuning with progressive unfreezing and Mixup. Full pretraining is not optional, and I will come back to that.

:::

## MCC Across Model Families

::: columns
:::: {.column width="54%"}

\centering
\includegraphics[width=\linewidth]{./images/transfer_mcc.pdf}
\small\color{gray}Prior (2,251) vs expanded (3,511) collection; labels are the difference

::::
:::: {.column width="44%"}

**Expanded collection, test MCC:**

- **TS-JEPA: 0.9194**
- LeJEPA: 0.9143
- CF-JEPA: 0.9083
- T-JEPA: 0.8982
- Mamba-3 TCN: 0.7816
- DeepStack: 0.7811

\vspace{0.3em}
Every world model clears the previous **0.756** Gaussian Process baseline by a wide margin.

::::
:::

::: notes

- Here is the headline. On the expanded collection the four world models take the top four places, from 0.9194 down to 0.8982 in MCC.
- The previous best result on this task was 0.756, from a Gaussian Process on hand-crafted features. We are not comparing against a weak baseline: the Gaussian Process was the winner of a 38-classifier study.
- The strongest supervised deep models, Mamba-3 TCN and DeepStack, land around 0.78. So the gap is not small, and it is not noise: three seeds, standard deviations of about 0.008.
- Green bars are the world models, orange are the deep supervised and state-space models. The hatched bars are the ones that degrade as the collection grows, which is the next slide.
- Note that the bars are on the log-ish scale of MCC, so a 0.14 gap is a large change in error structure, not a rounding difference.

:::

## Stability Under Collection Expansion

::: columns
:::: {.column width="53%"}

\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Prior} & \textbf{Expanded} & $\boldsymbol{\Delta}$ \\
\midrule
TS-JEPA & 0.8785 & \textbf{0.9194} & $+0.041$ \\
LeJEPA & 0.8850 & 0.9143 & $+0.029$ \\
CF-JEPA & 0.8661 & 0.9083 & $+0.042$ \\
T-JEPA & 0.8694 & 0.8982 & $+0.029$ \\
Mamba-3 TCN & 0.8844 & 0.7816 & $-0.10$ \\
DeepStack & 0.8810 & 0.7811 & $-0.10$ \\
SIGReg & 0.8858 & 0.4872 & $-0.40$ \\
HPO GRU & 0.8524 & 0.5746 & $-0.28$ \\
\bottomrule
\end{tabular}

::::
:::: {.column width="46%"}

**The contrast is the finding:**

- Self-supervised world models: **$+0.029$ to $+0.042$**
- Every deep supervised model: **degrades**, up to $-0.40$

\vspace{0.3em}
SIGReg's isotropic Gaussian prior is fitted to the training labels. A richer interference regime breaks it.

\vspace{0.3em}
\footnotesize Same device pool, so this measures **stability under collection expansion**, not transfer to a new environment.

::::
:::

::: notes

- This table is the scientific core of the paper, more than the headline number.
- Look at the direction of the arrows. All four self-supervised models _improve_ when the collection grows, by three to four hundredths of MCC. Every deep supervised and state-space model _degrades_, and SIGReg collapses from 0.8858 to 0.4872.
- Why? SIGReg's regulariser constrains the latent space around the training distribution, conditioned on the labels. When the interference regime becomes richer, those boundaries were fitted to artefacts of the old training set, and they break.
- The world models never optimise a label-conditioned boundary. They optimise prediction of the signal's own dynamics, so more data gives them more of the physics.
- One honest caveat, stated in the paper: the two collections share their device pool, so this is stability under collection expansion, not generalisation to an independent environment. The claim the table supports is the contrast between families, under one protocol.

:::

## Stacking and Calibration

\centering
\small
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lccccc}
\toprule
\textbf{Ensemble} & $\boldsymbol{n}$ & \textbf{Params} & \textbf{Prior} & \textbf{Expanded} & \textbf{Noise / Pure} \\
\midrule
21-member full stack & 21 & 74.8 M & 0.8873 & \textbf{0.9232} & \textbf{0.9563} / \textbf{0.9596} \\
16-member subset & 16 & 73.2 M & 0.8873 & \textbf{0.9232} & \textbf{0.9563} / 0.9179 \\
World-only stack & 7 & 6.68 M & 0.8872 & 0.9065 & 0.9450 / \textbf{0.9596} \\
Best-world subset & 3 & 3.42 M & 0.8784 & 0.9160 & 0.9341 / 0.8773 \\
Three-member Pareto & 3 & \textbf{1.26 M} & \textbf{0.8904} & 0.5753 & 0.9396 / 0.9179 \\
\bottomrule
\end{tabular}

\vspace{0.7em}
\begin{minipage}{0.95\linewidth}
\small
\begin{itemize}
\item Every member is \textbf{temperature-calibrated} on validation NLL, then fused by an L2 multinomial logistic-regression meta-learner.
\item The 16-member non-redundant subset matches the full 21-member stack exactly: the weak recurrent backbones add nothing.
\item Ensembles are pruned by a greedy \textbf{Pareto search}: MCC against parameter count.
\end{itemize}
\end{minipage}

::: notes

- Stacking the models does help, but the interesting part is where the gain stops.
- Every member is temperature-calibrated first: one scalar temperature per model, fitted on validation negative log-likelihood. This matters because uncalibrated softmax outputs are overconfident exactly under the interference we care about.
- The 21-member stack reaches 0.9232 on the expanded collection. On the multi-device subset alone it reaches 0.9563, and on pure single-device traces 0.9596.
- The 16-member non-redundant subset matches the full stack exactly: 0.9232. So the weak recurrent backbones contribute nothing.
- Now look at the last row. The Pareto-minimal ensemble from the _prior_ collection had 1.26 million parameters and 0.8904 MCC. On the expanded collection it collapses to 0.5753. Why? Because it contains SIGReg. The pruning was optimal for the old regime and wrong for the new one.
- That is the practical lesson: prune for stability, not only for size.

:::

## Accuracy versus Model Size

\centering

\vspace{0.6em}

\begin{minipage}{0.31\linewidth}
\centering
{\huge\bfseries 74.8\,M}\\[0.2em]
\footnotesize 21-member stack\\[0.1em]
\textbf{\large 0.9232} MCC\\[0.1em]
\scriptsize peak accuracy
\end{minipage}\hfill
\begin{minipage}{0.31\linewidth}
\centering
{\huge\bfseries 3.42\,M}\\[0.2em]
\footnotesize 3 world models\\[0.1em]
\textbf{\large 0.9160} MCC\\[0.1em]
\scriptsize $-0.007$ for $22\times$ fewer
\end{minipage}\hfill
\begin{minipage}{0.31\linewidth}
\centering
{\huge\bfseries 0.566\,M}\\[0.2em]
\footnotesize LeJEPA alone\\[0.1em]
\textbf{\large 0.9143} MCC\\[0.1em]
\scriptsize on 4 channels
\end{minipage}

\vspace{1.0em}

\begin{minipage}{0.95\linewidth}
\footnotesize
\textbf{Two deployment paths.} Rich 18-channel tokenisation needs the full hierarchy; a compact model designed for four kinematic channels nearly matches it, which is what makes an access-point implementation plausible.

\textbf{The trap:} the 1.26 M three-member Pareto subset collapses to 0.5753 on the expanded collection, because it contains the supervised SIGReg.
\end{minipage}

::: notes

- Let me put the parameter question in perspective.
- The full stack needs 74.8 million parameters and gives 0.9232.
- A three-member subset made only of world models needs 3.42 million and gives 0.9160. That is seven thousandths of MCC for twenty-two times fewer parameters.
- And LeJEPA alone, with 0.566 million parameters over only four channels, gives 0.9143. Essentially the same accuracy as the ensemble, at a hundred and thirty times fewer parameters than the stack.
- That is not a small engineering detail: it is the difference between a datacentre model and something that runs on the access point you already have on the bus.
- The trap on the bottom is worth remembering: optimising for size on one collection does not survive a change in regime, because the Pareto frontier inherited a model that collapses.

:::

## Channel-Level Attribution

::: columns
:::: {.column width="55%"}

\centering
\includegraphics[width=0.86\linewidth]{./images/feature_importance_18ch.pdf}

\small\color{gray}Mean absolute SHAP value per channel, TS-JEPA, 3,511-sample test split

::::
:::: {.column width="46%"}

\footnotesize

\textbf{What drives the prediction:}

- \textbf{Peak location} ($6.04\times10^{-3}$): \emph{when} the door transition happens
- \textbf{STFT band 0} ($5.97\times10^{-3}$): coarse Doppler of physical motion
- \textbf{FFT DC / AC} ($4.40$ / $3.99\times10^{-3}$): energy shift between zones
- \textbf{Everything else} ($1.22$ kurtosis, $0.95$ std, high STFT): multipath, not motion

\vspace{0.4em}

\textbf{What a bar means.} A Shapley value is a channel's average marginal contribution over every coalition of the other seventeen; the bars are the mean $|\phi_j|$ over the test split (Lundberg \& Lee, NeurIPS 2017).

\vspace{0.2em}
\color{gray}
Next slide: the same question, under controlled architectures.

::::
:::

::: notes

- Shapley Additive exPlanations, from cooperative game theory: Lundberg and Lee, NeurIPS 2017. A Shapley value is a player's average marginal contribution over every possible coalition of the other players; here the players are the eighteen channels, and the bars are the mean absolute value over the test split.
- The decomposition is exact: the eighteen attributions add up to the model output minus its expected value, so the bars account for the whole prediction, not a subset of it.
- Attributions are computed on the leading TS-JEPA model over the 18-channel hierarchy.
- Two features dominate: the peak location index, which is essentially the timestamp of the door transition, and the lowest STFT band, which captures the coarse Doppler shift of a body in motion.
- Then come the DC and AC energies of the rolling FFT: those are the baseline energy shift as the device moves between two zones with different attenuation. That is physics, and it is what the model uses.
- What sits at the bottom? Kurtosis, standard deviation, the higher STFT bands. Those describe multipath scattering, which is exactly the nuisance we want the model to ignore. And it mostly does.
- This is a nice independent confirmation of the world-model argument: the useful signal is macroscopic motion shape, not fine-grained radio texture.

:::

## Feature-Channel Ablation

\centering
\scriptsize
\setlength{\tabcolsep}{6pt}
\begin{tabular}{lcccc}
\toprule
\textbf{Feature configuration} & $\boldsymbol{C}$ & \textbf{TS-JEPA} & \textbf{CF-JEPA} & \textbf{T-JEPA} \\
\midrule
Full 18-channel hierarchy & 18 & \textbf{0.9194} & \textbf{0.9083} & \textbf{0.8982} \\
Top-12 SHAP channels & 12 & 0.5815 & 0.8098 & 0.5856 \\
Base kinematics (raw, $\Delta$, $\Delta^2$, dev) & 4 & 0.4486 & 0.4366 & 0.4787 \\
Rolling statistics (moments, range) & 5 & 0.3447 & 0.4351 & 0.4944 \\
Spectral domain (FFT, STFT 0--3) & 6 & 0.2981 & 0.1869 & 0.2283 \\
Raw RSSI only (single channel) & 1 & 0.2431 & 0.3305 & 0.3395 \\
\bottomrule
\end{tabular}

\vspace{0.7em}
\begin{minipage}{0.95\linewidth}
\footnotesize
Restricting a fixed architecture to any single functional domain destroys it: TS-JEPA falls from 0.9194 to 0.4486 on the four base channels, and to 0.2431 on raw RSSI. Even the twelve most important channels, selected by SHAP, are not enough: the synergy is not additive.
\end{minipage}

::: notes

- This is the ablation that surprised us, and it is the strongest evidence for the world-model framing.
- If you keep the architecture identical and only change the input, removing channels is catastrophic. TS-JEPA goes from 0.9194 with eighteen channels, to 0.4486 with the four base kinematic channels, to 0.2431 with raw RSSI alone.
- The interesting row is the second: the top twelve channels ranked by SHAP give 0.5815. So it is not that twelve channels are missing some information; it is that a patch tokenizer over a rich set learns cross-channel interactions that a reduced set cannot express.
- Consequence for deployment: you cannot simply strip a rich model down. Either you keep the hierarchy, or you use an architecture designed for a small input, like LeJEPA, which gets 0.9143 on four channels with 0.566 million parameters.
- In other words, the two deployment paths are real, and the middle ground is the worst of both.

:::

## Privacy Through Edge Inference

::: columns
:::: {.column width="48%"}

**The threat is real:**

- Passive sniffing needs no cooperation and no visible trace
- Radio sequences survive MAC randomisation
- Commuter regularity links a passenger across days
- Profiling defeats naive anonymisation and DP (Silva et al.)

::::
:::: {.column width="48%"}

**What the architecture buys:**

- 0.9143 MCC at **0.566 M** parameters runs _inside_ the AP firmware
- Raw RSSI and latents live in RAM and are discarded
- Only aggregated counts leave: $\Delta$boarding, $\Delta$alighting

\vspace{0.3em}
\footnotesize The accuracy that makes the sensing useful is the same property that makes edge confinement possible.

::::
:::

::: notes

- Passive sensing avoids cameras and microphones, but it is not automatically private. Any receiver in range can capture the same frames we do, and it can do so without the operator knowing.
- The literature is clear that anonymisation is not enough. Silva and colleagues showed that profiling models recover behavioural patterns from interaction data, and that differential privacy can be defeated when a classifier exploits temporal correlations. In transit, the regular commute is exactly such a correlation: the same person, the same door, the same time of day.
- Our mitigation is architectural rather than legal. Because LeJEPA reaches 0.9143 with 0.566 million parameters, inference can run inside the access point. The raw sequence never leaves the vehicle, the latents never leave the process, and only boarding and alighting counts go to the backend.
- That does not make the system unharmful, but it removes the centralised corpus that makes longitudinal profiling cheap.
- The future work in the paper goes further: elastic privacy, throttling sensing resolution by the vehicle's motion state.

:::

## What These Results Mean

::: columns
:::: {.column width="48%"}

**What we have demonstrated:**

- **Four movement classes** from 10 RSSI values, no dedicated hardware
- **0.9194** single model · **0.9232** calibrated ensemble
- **0.9563** under multi-device interference
- **Stability**: world models gain as data grows, supervised models break
- **0.9160** at 3.42 M parameters, edge-feasible

::::
:::: {.column width="48%"}

**What we must be careful about:**

- Two **controlled indoor zones**, not a vehicle in service
- Seven phone models, no demographic attributes recorded
- Labels from the **session protocol**, not independent annotation
- Training corpus **overlaps the evaluation splits**, so absolute values are optimistic
- 160 single-device traces are too few for a separate held-out pure evaluation

::::
:::

::: notes

- Let me be explicit about the boundary between what we proved and what we hope.
- We have shown that a self-supervised world model can classify four passenger movement states from commodity RSSI, above 0.92 MCC, with an edge-feasible compact variant, and that it stays stable when the collection grows while supervised baselines do not.
- We have not shown that this works on a moving vehicle. Our environment is two controlled indoor zones. A real bus adds vibration, metal reflections, crowding and body absorption, all of which change the RSSI distribution.
- The device pool is seven phone models, and we recorded no participant demographics, so we cannot claim equivalence across passenger groups.
- Most important, and stated in the paper: the training corpus overlaps the evaluation splits. The absolute values are therefore optimistic. What the tables support is the comparison between model families, measured on identical splits under one protocol.
- Those limitations are not fatal, they define the next study.

:::

## Conclusions and Future Work

::: columns
:::: {.column width="45%"}

**This work:**

- **Self-supervised world models** for passive ISAC in transit
- Four JEPA variants, 18-channel hierarchy, calibrated ensembles
- **0.9194** single · **0.9232** ensemble · **0.9160** at 3.42 M
- **Stability under collection expansion**, where supervised models collapse

\vspace{0.4em}
\footnotesize Models and code: github.com/ATNoG/inMotion

::::
:::: {.column width="47%"}

**Next steps:**

- **Cooperative multi-AP fusion** with spatial diversity
- **Online continual adaptation** to seasonal radio drift
- **Elastic privacy**: throttle resolution by motion state
- **Operational validation** on a vehicle in service
- Close the gap between single models (0.9194) and the 0.9563 ensemble result on the multi-device subset

::::
:::

::: notes

- To conclude. We reformulated passenger movement classification as learning a world model of the radio channel, and it beats the supervised state of the art on this task by a wide margin: 0.9194 for a single model, 0.9232 for a calibrated ensemble, 0.9160 for a three-member subset of 3.42 million parameters.
- The result I would like you to remember is not the peak number but the direction of the arrows: as the collection grows, the self-supervised models improve and the supervised ones break. That is what a predictive objective over the signal buys you.
- The limitations are the agenda. We need multi-AP spatial diversity to resolve overlapping zones. We need online adaptation for seasonal drift. We need a real vehicle, in service, with energy and thermal measurements.
- Elastic privacy is the part I am most excited about: sensing resolution that depends on what the vehicle is doing, fine only at the doorway during a dwell, coarse while cruising.
- The code and the models are on GitHub. Thank you.

:::

## Acknowledgements

\centering

\vspace{1em}

**This work is supported by the European Regional Development Fund (FEDER)**

through the Regional Operational Programme of Centre (CENTRO 2030)

of the Portugal 2030 framework

\vspace{0.5em}

Project **inMotion**, Nr. 21359 (CENTRO2030-FEDER-02225900)

\normalsize Questions?

\vspace{0.8em}

\small \color{gray} Code and models: github.com/ATNoG/inMotion · Paper: FNWF 2026

::: notes

- Before questions, the funding acknowledgement: this work is part of inMotion, supported by European structural funds through CENTRO 2030.
- The models, the training scripts and the evaluation harness are public on GitHub, so every number in this talk can be reproduced.
- I am happy to take questions.

:::
