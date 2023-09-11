import inspect

def dbe(*args):
    frame = inspect.currentframe().f_back
    Str = inspect.getframeinfo(frame).code_context[0].strip()

    var_names = Str[Str.find("(")+1:-1].split(",")

    for var_name, arg in zip(var_names, args):
        print(f"{var_name.strip()} = {arg}")
        
    exit()