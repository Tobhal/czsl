from num2words import num2words
from typing import Dict, List

def generate_ordinal_number_dict(upper_limit: int) -> Dict[int, str]:
    """
    Generate a dictionary of ordinal numbers up to the given upper limit.

    Parameters:
    - upper_limit (int): The upper limit for generating ordinal numbers.

    Returns:
    - Dict[int, str]: A dictionary where keys are integers and values are their corresponding ordinal strings.
    """
    ordinal_dict = dict()

    for number in range(1, upper_limit + 1):
        ordinal_representation = num2words(number, ordinal=True)
        ordinal_dict[number] = ordinal_representation

    return ordinal_dict

def generate_clip_text(upper_limit: int, shapes: List[int]) -> str:
    """
    Generate a string describing the shapes using ordinal numbers.

    Parameters:
    - upper_limit (int): The upper limit for generating ordinal numbers.
    - shapes (List[int]): A list of shape values.

    Returns:
    - str: A string describing the shapes using ordinal numbers.
    """
    ordinal_numbers = generate_ordinal_number_dict(upper_limit)
    clip_text = ''

    for number, ordinal in ordinal_numbers.items():
        clip_text += f'The {ordinal} shape is {shapes[number - 1]}\n'

    return clip_text


if __name__ == '__main__':
    # Shapes form phos
    shapes = [2, 2, 2, 3, 0, 3, 1, 0, 0, 0, 0, 0, 0]

    clip_string = generate_clip_text(10, shapes)
    print(clip_string)