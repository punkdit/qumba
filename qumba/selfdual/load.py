#!/usr/bin/env python

# See:
# https://www.math.is.tohoku.ac.jp/~munemasa/research/codes/sd2.htm


def get_items(name = "24-II.magma"):

    if "-" in name:
        stem = name.split("-")[0]
    else:
        stem = name.split(".")[0]
    n = int(stem)

    import pathlib
    path = pathlib.Path(__file__).parent.resolve()
    path = pathlib.Path(path, name)
    
    s = open(path).read()
    
    s = s.replace(';', '')
    lines = s.split("\n")
    lines = [l for l in lines if "codes" not in l]
    lines = [l for l in lines if "PowerStructure" not in l]
    
    s = "[" +  "\n".join(lines)
    s = s.replace("LinearCode<GF(2),%d|"%n, "[")
    s = s.replace("LinearCode<GF(2), %d |"%n, "[")
    s = s.replace("GF(2)|", "")
    s = s.replace("GF(2) |", "")
    s = s.replace(">", "]")

    assert "GF" not in s
    assert "LinearCode" not in s
    
    f = open("dump.py", "w")
    print(s, file=f)
    
    s = s.replace("\n", " ")
    
    try:
        items = eval(s)
    except SyntaxError:
        print("Parse failed, see dump.py")
        return

    #print(items)

    return items


