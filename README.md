# FINANCIAL_KG_DS

Welcome to the FINANCIAL_KG_DS repository. This project focuses on creating and managing a financial knowledge graph and data science workflows.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
The FINANCIAL_KG_DS project aims to provide tools and resources for building and analyzing financial knowledge graphs. It includes data processing scripts, graph algorithms, and visualization tools.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ondrabimka/FINANCIAL_KG_DS.git
cd FINANCIAL_KG_DS
pip install -r requirements.txt
```

To have the same data make sure you have https://github.com/ondrabimka/FINANCIAL_KG running.
Create .env file in the root directory and add the following environment variables:

```bash
DATA_PATH=''
```

The data should point to FINANCIAL_KG with specific date:
example: ".../FINANCIAL_KG/data/data_2024-09-06"

## Usage
Training scripts are currently located in [train location](financial_kg_ds/train/README.md) directory.