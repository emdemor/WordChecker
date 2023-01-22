# WordChecker

The WordChecker class is used to check the existence of a word in a dictionary of a certain language and find similar words.

## Installation

```
pip install WordChecker
```


## Usage

```python
from WordChecker import WordChecker

# Initialize the class with the desired language
word_checker = WordChecker("pt")

# Check if a word exists in the dictionary
word_checker("palavra")

# Find the most similar word to the one passed as a parameter
word_checker.most_similar_word("palavra")

# Find the n most similar words to coorect the one passed as a parameter.
word_checker.correct_word("palavra", n=3, cutoff=0.6)

# Find the n most similar words to the one passed as a parameter
word_checker.most_similar_n_words("palavra", n=3, cutoff=0.6)
```

## Parameters

- **lang** (str): Language chosen for the dictionary.

## Attributes

- **nlp** (spacy.language.Language): Object from the spacy package with the information of the chosen language.
- **frequency** (dict): Dictionary with the frequency of words in the chosen language.
- **vocabulary** (array): List of words present in the dictionary of the chosen language.

## Methods

- **__call__(word: str) -> bool**: Check if a word exists in the dictionary.
- **most_similar_word(word: str) -> str**: Find the most similar word to the one passed as a parameter.
- **correct_word(word: str, n: int = 3, cutoff: float = 0.6) -> np.ndarray**: Find the n most similar words to coorect the one passed as a parameter.
- **most_similar_n_words(word: str, n: int = 3, cutoff: float = 0.6) -> np.ndarray**: Find the n most similar words to the one passed as a

## Contributing

1. Fork it (<https://github.com/your-github-user/WordChecker/fork>)
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.