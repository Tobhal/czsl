import inspect

def dbe(*args, should_exit=True):
    """Print names and values of input variables. Exit the program if should_exit is True."""

    if not args:
        raise ValueError("Function requires at least one argument")

    frame = inspect.currentframe().f_back

    try:
        code_string = inspect.getframeinfo(frame).code_context[0].strip()
    except IndexError:
        raise ValueError("Function can only be called with variable arguments")

    # Parse out actual variable names
    var_names = code_string[code_string.find("(")+1:-1].split(",")

    for var_name, arg in zip(var_names, args):
        var_name_strip = var_name.strip()
        if isinstance(arg, list):
            print(f"{var_name_strip}({len(arg)}) = {arg}")
        elif isinstance(arg, dict):
            print(f"{var_name_strip}(keys:{len(arg.keys())}) = {{")
            for k, v in arg.items():
                print(f"\t{k} : {v}")
            print("}")
        elif isinstance(arg, set):
            print(f"{var_name_strip}({len(arg)}) = {arg}")
        elif isinstance(arg, tuple):
            print(f"{var_name_strip}({len(arg)}) = {arg}")
        else:
            print(f"{var_name_strip} = {arg}")

    if should_exit:
        exit()

if __name__ == '__main__':
    test_var = 'test'

    test_array = [1,2,3,4,5,6]

    test_dict = {
        "key1": "value1",
        "key2": "value2",
        "key3": ["value3a", "value3b", "value3c"],
        "key4": {"nested_key1": "nested_value1", "nested_key2": "nested_value2"}
    }

    test_set = {"apple", "banana", "cherry"}

    test_tuple = ("apple", "banana", "cherry")

    dbe(test_var, test_array, test_dict, test_set, test_tuple)