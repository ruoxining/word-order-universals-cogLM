import sys

actions = sorted(list(set([w + " 100" for line in sys.stdin for w in line.split() if w[0].isupper()])))
print("\n".join(actions))
    