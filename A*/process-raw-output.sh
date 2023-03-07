#!/bin/bash

if [ "$#" -ne 2 ] ; then
    echo "Usage: $0 input-file.txt output-file.csv"
    echo "Also pls be sure to preprocess the input file for weird exceptions saying During..."
    exit 1
fi

IN="$1"
OUT="$2"

sed -Ee '/^regex 106$/d' -e '/^text 97$/d' "$IN" | sed -Ee '
:x
N
/\n$/!bx
s/.*MemoryError.* ([^ ]*)elapsed.* ([^ ]*)maxresident.*/0.0,"\1",\2,"MEMORY"/
s/.*non-zero status 124.* ([^ ]*)elapsed.* ([^ ]*)maxresident.*/0.0,"\1",\2,"TIME"/
s/.*No regex works.* ([^ ]*)elapsed.* ([^ ]*)maxresident.*/0.0,"\1",\2,"INCOMPLETE"/
s/([^ ]*) ([^\n]*).* ([^ ]*)elapsed.* ([^ ]*)maxresident.*/\1,"\3",\4,"\2"/ ; s/\\/\\\\/g
' > "$OUT"

python - <<EOS
import ast
from human_examples import *

def strip_suffix(x, s):
    return x[0:len(x)-len(s)] if x[-len(s):] == s else x

ls = [l.rstrip().split(',', 3) for l in open("$OUT")]
for (__, r), l in zip(all_benchmarks, ls):
    l[3] = repr(strip_suffix(ast.literal_eval(l[3]), ' '+r))

out = open("$OUT", 'w')
out.writelines([','.join(l) + ',' + repr(r) + ',' + str(len(ex)) + '\n' for l, (ex,r) in zip(ls, all_benchmarks)])
EOS


