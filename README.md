# Meta-Analysis of Blood Glucose Prediction Models and Methods for Meal and Physical Activity Detection in the Context of Diabetes Mellitus
This GitHub repository contains the code and data related to the academic article *Meta-Analysis of Technologies for Diabetes Treatment: Glycemic Control, Prediction, Meal and Physical Activity Detection*, which is currently in works. Once published, the necessary links to the article will be added.

[![CC BY-NC 4.0][cc-by-nc-shield]][cc-by-nc]

[![CC BY-NC 4.0][cc-by-nc-image]][cc-by-nc]

[cc-by-nc]: https://creativecommons.org/licenses/by-nc/4.0/
[cc-by-nc-image]: https://licensebuttons.net/l/by-nc/4.0/88x31.png
[cc-by-nc-shield]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg

## How to Use
To use this code, you first need to install Python on your computer and optionally some IDE in which you can comfortably interact with the scripts. Specifically, I have used **Python version 3.11.4** and the following libraries:
| LIBRARY      | VERSION     |
| ------------ | ----------- |
| matplotlib   | 3.7.2       |
| numpy        | 1.25.2      |
| pandas       | 2.1.0       |
| scikit-learn | 1.3.0       |
| scipy        | 1.11.2      |

With everything prepared, the code should be ready to use.

## Description of Files
The repsitory contains the following files:
1) `plots.py` – implements the meta-analysis, i.e., creating the plots and printing reported values;
2) `algorithms.csv` – data related to the Time-Shift, Pattern Prediction, and BGLP 2025 winnter models;
3) `meal.csv` – data related to the meal detection models;
4) `physicalactivity.csv` – data related to the physical activity detection models;
5) `prediction.csv` – data related to the blood glucose prediction models;
