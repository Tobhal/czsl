from modules.utils import generate_label_for_description, generate_phoc_vector
from utils.dbe import dbe
from typing import Union
from num2words import num2words

def split_string_into_chunks(input_string, chunk_size: int):
    """
    Split the input string into chunks of 'chunk_size' characters.

    Args:
    - input_string (str): The string to be split.
    - chunk_size (int): The maximum number of characters in each chunk.

    Returns:
    - list of str: A list containing the split substrings.
    """
    # Use a list comprehension to split the string into chunks of 'chunk_size' characters
    return [input_string[i:i+chunk_size] for i in range(0, len(input_string), chunk_size)]


def gen_label_description(label: Union['phos', 'phoc'], name: str = '') -> str:
    description = ''

    for idx, level in enumerate(label):
        description += f'{name}{idx} '

        for i, split in enumerate(level):
            # description += ''.join([str(int(x)) for x in split[0]])

            for shapes in split:
                for l, s in enumerate(shapes):
                    shape_text = num2words(int(s))
                    description += f'shape{i}_{shape_text} '

            # if i != len(level) - 1:
                # description += ' '
        
        # if idx != len(label) - 1:
            # description += '\n'

    dbe(description)

    return description

def get_phosc_description(word: str) -> str:
    phos = generate_label_for_description(word, 3)
    phoc = generate_phoc_vector(word)

    description = gen_label_description(phos)
    # description += '\n'
    # description += gen_label_description(phoc, 'phocLevel')
    return description
