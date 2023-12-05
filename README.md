# NBA Rookies Career Longevity Prediction

<img src="https://github.com/titoausten/NBA-Career-Longevity-Prediction-/blob/main/download.png" />

## Overview

<p>

The National Basketball Association (NBA) is a professional basketball league in North America. The league comprises 30 teams (29 in the United States and 1 in Canada) and is one of the four major professional sports leagues in the United States and Canada. It is the premiere men's professional basketball league in the world. [Wikipedia](https://en.m.wikipedia.org/wiki/National_Basketball_Association).

</p>

<p>

Career longevity is dependent on various factors for any player in all the games and so for NBA Rookies. The factors like games played, count of games played, and other statistics of the player during the game.

</p>

<hr>

## Table of Contents
- [Objective](#objective)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [License](#license)

## Objective

 <p>

Using machine learning techniques to determine if a player’s career will flourish or not.

</p>

<hr>

## Dataset

<p>

The dataset contains player statistics for NRB Rookies. There are 1100+ observations in the train dataset with 19 variables excluding the target variable (i.e. Target).

</p>

### Data Description

<p>

* GP: Games played

The values for given attributes are averaged over all the games played by players.

* MIN:  Minutes Played

* PTS: Number of points per game

* FGM: Field goals made

* FGA: Field goals attempt

* FG%: field goals percent

* 3P Made: 3 point made

* 3PA: 3 points attempt

* 3P%: 3 point percent

* FTM: Free throw made

* FTA: Free throw attempts

* FT%: Free throw percent

* OREB: Offensive rebounds

* DREB: Defensive rebounds

* REB: Rebounds

* AST: Assists

* STL: Steals

* BLK: Blocks

* TOV: Turnovers

* Target: 0 if career years played < 5, 1 if career years played >= 5

</p>
<hr>

## Requirements

To run this project, you can run:
```
pip install -r requirements.txt

```

## Usage
1. Clone this repository to your local machine:
```
git clone https://github.com/titoausten/Career-Longevity-Prediction-.git
```

2. Setup and preprocess data:
```
python src/setup_data.py
```

3. train the models:
```
python src/train.py
```

4. Evaluate the models:
```
python src/predict.py
```

5. Predict new data:
```
python src/predict.py
```

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
