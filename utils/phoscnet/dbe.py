import inspect

def dbe(*args):
    frame = inspect.currentframe().f_back
    Str = inspect.getframeinfo(frame).code_context[0].strip()

    var_names = Str[Str.find("(")+1:-1].split(",")

    for var_name, arg in zip(var_names, args):
        # Checking if the argument is a list
        if isinstance(arg, list):
            print(f"{var_name.strip()}({len(arg)}) = {arg}")
        else:
            print(f"{var_name.strip()} = {arg}")

    exit()
    
if __name__ == '__main__':
    var = 'variable'
    vec = [1, 2, 3]
    dbe(var, vec)