---
layout: default
title: Projects
permalink: /projects/
---

# Digital mirror universe: advancing the digital twin framework for multiscale modeling
## Using EHR claims data as an example

### Introduction

A digital twin in healthcare research is a dynamic virtual model of an individual patient or biological system, created by integrating clinical, biological, and sensor data with computational simulations. Continuously updated to reflect a patient’s real-time state, it allows researchers to model disease progression, test interventions in silico, and predict treatment responses. This approach supports precision medicine by enabling more personalized and data-driven healthcare decisions.

An electronic health record (EHR) systematically documents a patient’s diagnostic history, procedures, prescriptions, and other clinical information, reflecting their health status as it evolves over time and across care settings. This study aims to develop a model that encodes a subset of a patient's EHR, represented by diagnosis codes (claims), into a stable, time-invariant vector embedding. This embedding serves as a latent digital representation of the patient's underlying health state. Collectively, these embeddings form a structured space that represents the broader patient population, extending the concept of a digital twin. We refer to this as a **digital mirror universe**: a virtual representation of the entire population that can be used for multiple tasks, including disease risk prediction, missing-data imputation, synthetic EHR generation, and downstream integration with genetic and other biomedical data.

A diagnosis EHR, which we refer to as a "sentence", is composed of tens of repeating segment (a "word") <pre>Diagnosis code:Age:Time</pre>. It worth noting some characteristics of the medical code sentence:
