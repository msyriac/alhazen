


def get_patch_degrees(Config,section):

    try:
        patch_arcmins = Config.getfloat(section,"patch_arcmins")
        arcmin = True
    except:
        arcmin = False

    try:
        patch_degrees = Config.getfloat(section,"patch_degrees")
        degree = True
    except:
        degree = False


    if arcmin and degree:
        raise ValueError
    elif arcmin:
        return patch_arcmins/60.
    elif degree:
        return patch_degrees
    else:
        print "ERROR: Patch width not specified."
        sys.exit()

    
