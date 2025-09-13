import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if root not in sys.path:
    sys.path.insert(0, root)
src = os.path.join(root, 'src')
if src not in sys.path:
    sys.path.insert(0, src)
