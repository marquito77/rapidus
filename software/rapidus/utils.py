def cfgGetVal(cfgfile, section, value):
    secStr = "[{}]".format(section)
    with open(cfgfile, "r") as fh:
        inSec = False
        for line in fh:
            line = line.strip()
            if line.startswith("["):
                if line == secStr:
                    inSec = True
                else:
                    inSec = False
            if inSec and line.startswith(value):
                toks = line.split("=")
                if len(toks) != 2:
                    print("Error parsing cfgfile {}: Could not parse string {}".format(cfgfile, line))
                    return None
                vals = toks[1]
                vals = vals.strip()
                valToks = vals.split(",")
                ret = []
                for valStr in valToks:
                    try:
                        val = int(valStr)
                    except:
                        val = float(valStr)

                    ret.append(val)
                if len(ret) == 1:
                    ret = ret[0]
                #print("cfgGetVal(): {}/{} = {}".format(section, value, ret))
                return ret
    print("Error: Could not find section/val ({}/{}) in file {}".format(section, value, cfgfile))
    return None
