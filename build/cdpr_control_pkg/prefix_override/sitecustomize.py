import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/alex/Greenwall/Greenwall-Pruning/install/cdpr_control_pkg'
