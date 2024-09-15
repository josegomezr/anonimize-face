def do_patch():
    import os

    has_ld_path = len(os.environ.get('LD_LIBRARY_PATH', '')) > 0

    ld_path = "{}/cudnn/lib".format(nvidia.__path__[0])
    old = os.environ['LD_LIBRARY_PATH'] 

    if has_ld_path:
        ld_path += ":"
    ld_path += old

    print('export LD_LIBRARY_PATH={}'.format(ld_path))

try:
    import nvidia
    do_patch()
except ImportError:
    pass
    
