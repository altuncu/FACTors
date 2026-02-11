# FACTors: A New Dataset for Studying Fact-checking Ecosystem

This repository includes the dataset and source codes presented in the paper titled *"FACTors: A New Dataset for Studying Fact-checking Ecosystem"* accepted for [*the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025)*](https://sigir2025.dei.unipd.it/) as a Resource & Reproducibility paper.

*FACTors* contains 118,112 claims from 117,993 fact-checking reports in English (co-)authored by 1,953 individuals and published during the period 1995-2025 by 39 fact-checking organisations that are active signatories of the [IFCN (International Fact-Checking Network)](https://ifcncodeofprinciples.poynter.org/signatories) and/or [EFCSN (European FactChecking Standards Network)](https://members.efcsn.com/signatories). It contains 7,327 overlapping claims investigated by multiple fact-checking organisations, corresponding to 2,977 unique claims.

# Repository Content

This repository contains the following:

- ```FACTors.py```: The file containing a class for the dataset, which can be used to generate the dataset from raw scraped data and calculate some useful statistics
- ```/data```: The dataset file in CSV format with two additional files with author and organisation statistics derived from the dataset
- ```/docs```: Figures generated during the study, most of which were included in the paper
- ```/scripts```: Source codes of the experiments using the dataset
  - ```/construction```: Codes used for constructing the dataset
  - ```/experiments```: Codes of the experiments mentioned in the paper

In addition, the Apache Lucene (version 8.11.0) index for *FACTors* can be found [here](https://drive.google.com/file/d/1PRgV7jpGt7IykhE2_pVjltP34DQD5C4_/view?usp=drive_link).

# Dataset Structure

FACTors dataset is provided with a single CSV file, [FACTors.csv](https://github.com/altuncu/FACTors/blob/main/data/FACTors.csv), which contains the following columns:

| Field Name | Description
| -----------|----------------
| Row ID | Primary Key
| Report ID | ID given to each unique report
| Claim ID | ID given to each unique claim
| Claim | Textual claim being fact-checked
| Content | *(not published to prevent copyright infringement)*
| Date published | Date of publication of the report in ISO 8601 format
| Author | Author(s) of the fact-checking report
| Organisation | Name of the fact-checking organisation publishing the report
| Original verdict | Conclusion of the fact-check
| Title | Heading of the report
| URL | Online link to the report
| Normalised rating | One of six predefined ratings derived from the original verdict, using a fine-tuned RoBERTa [model](https://huggingface.co/ealtuncu/verdict-normaliser-roberta)

Along with the dataset, two more CSV files ([author_stats.csv](https://github.com/altuncu/FACTors/blob/main/data/author_stats.csv) and [org_stats.csv](https://github.com/altuncu/FACTors/blob/main/data/org_stats.csv)) are provided, containing the following statistics for each author and fact-checking organisation covered, respectively:

- **Fact-checking experience**: Time difference between a fact-checker's first and last published fact-checking report
- **Number of fact-checks**: Total number of fact-checking reports a fact-checker has published
- **Percentage of unique fact-checks**: Proportion of fact-checked claims that have not been investigated by any other fact-checker previously
- **Fact-checking rate**: Mean and standard deviation of how frequently a fact-checker publishes fact-checking reports
- **Number of authors**: Number of authors each organisation has employed
- **Number of organisations**: Number of fact-checking organisations each author has published with
- **Word count**: Mean and standard deviation of the number of words in the fact-checking reports published by a fact-checker

## Contact

If you have any questions about the dataset and/or source codes, please reach out to the contributors via email:
- For more information about FACTors and its example applications provided: [Enes Altuncu](mailto:enes.altuncu@iuc.edu.tr)
- For the Apache Lucene index version of FACTors: [Dwaipayan Roy](mailto:dwaipayan.roy@iiserkol.ac.in)

## Citation

Please cite our work as follows if you use our dataset or the provided source codes in your research:

````
@inproceedings{FACTors2025,
  title={{FACTors}: A New Dataset for Studying Fact-checking Ecosystem},
  authors={Altuncu, Enes and 
           Ba\c{s}kent, Can. and 
           Bhattacherjee, Sanjay and 
           Li, Shujun and 
           Roy, Dwaipayan},
  year={2025},
  numpages={10},
  doi={10.1145/3726302.3730339},
  booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '25), July 13--18, 2025, Padua, Italy},
  publisher={ACM},
}
````
