def lesion_parser(code):
    if str(code == 0):
        return (0,0,0,0)
    else:
        if (code.startswith('11') or code.startswith('00')) and len(code) == 5:
            rest = code[2:]
            loc = code[0:2]
        else:
            rest = code[1:]
            loc = code[0:1]

        type = rest[0]
        subtype = rest[1]
        specific = rest[2]
        return (loc, type, subtype, specific)