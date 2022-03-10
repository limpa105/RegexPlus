#!/usr/bin/sed -nEf
1s/.*/    Example(/p
s/^"(.*)",\+$/        "\1",/p
$s/.*/        "&"),\n/p

