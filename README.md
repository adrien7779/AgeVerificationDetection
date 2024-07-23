# Popup Age Verification Detection

## Overview

The code provides an HTML extraction function that accesses a specified number of domains from the Tranco library. It processes the HTML of each page using Beautiful Soup and Playwright to scrape them. Some domains may return errors (as they don't exist), but these errors are logged into the database.

Text within the site's HTML is extracted and passed to a pretrained DistilBERT model to analyze the presence of age verification language. Detected information is then stored in an SQLite database.

Please note that the model may give false positives when encountering new text patterns, text in other languages, or ambiguous text. Such cases should be reviewed and used to update the training data for future model improvements.

## Training a New Model

To train a new model with different parameters or datasets, follow these steps:
1. Optionally adjust the training data in `data_preparation.py`
2. Rename the model file in `main.py` to avoid overwriting existing models.
3. Adjust hyperparameters in `main.py`, ensuring a small learning rate to maintain DistilBERT's original training.
4. Modify batch size and other parameters in `train.py`.
5. Optionally, retrain an existing model by setting `reTrain=True` in `main.py`.
6. Consider using the weighted cross-entropy loss function from `model.py` to tune model performance.

Training and inference are resource-intensive tasks, best performed using GPUs. On Windows systems, Docker is required for setup. A tutorial can be found [here](https://www.youtube.com/watch?v=YozfiLI1ogY) for detailed instructions.

## Second Pass with Whois Data

The host finding code (`whois.py`) is not integrated directly into the `htmlExtract` function to optimize runtime. Run this code separately as a second pass to populate the database, utilizing the Whois library for server queries.

## Considerations

- Ensure the playwright scraper detects and translates non-English sites before extraction to maximize detection capabilities.
- Update training data to address false positives encountered during detection.
- Develop a method to categorize sites, possibly integrating with external datasets like the Kaggle site categorization Excel file (currently incomplete and outdated).





