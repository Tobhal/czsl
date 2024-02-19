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

