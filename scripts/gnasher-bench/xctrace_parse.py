#!/usr/bin/env python3
"""Parser for `xctrace export` XML tables. Resolves id/ref deduplication (ids may be defined at any
nesting depth) and returns rows as ordered lists of (tag, fmt, text). to_ns() parses durations like
"4.12 µs" and timestamps like "00:00.763.750"."""
import re
UNIT={'ns':1,'µs':1e3,'us':1e3,'ms':1e6,'s':1e9}
_open=re.compile(r'<([a-z0-9-]+)((?:\s+[a-z0-9-]+="[^"]*")*)\s*(/?)>')
_elem=re.compile(r'<([a-z0-9-]+)((?:\s+[a-z0-9-]+="[^"]*")*)\s*(?:/>|>(.*?)</\1>)', re.S)
def to_ns(fmt, text=''):
    if text and text.strip().lstrip('-').isdigit():
        return int(text)
    m=re.match(r'\s*([\d.]+)\s*(ns|µs|us|ms|s)\b', fmt or '')
    if m:
        return int(float(m.group(1))*UNIT[m.group(2)])
    m=re.match(r'\s*(\d+):(\d+)\.(\d+)\.(\d+)', fmt or '')
    if m:
        mm,ss,ms,us=map(int,m.groups())
        return ((mm*60+ss)*10**9)+ms*10**6+us*10**3
    return 0
def rows(path):
    xml=open(path).read()
    ids={}
    for m in _open.finditer(xml):
        attrs=m.group(2); idm=re.search(r'\bid="(\d+)"', attrs)
        if idm:
            fmt=re.search(r'\bfmt="([^"]*)"', attrs)
            text=''
            if fmt is None and not m.group(3):
                t=re.match(r'([^<]*)<', xml[m.end():]); text=t.group(1).strip() if t else ''
            ids[idm.group(1)]=(m.group(1), fmt.group(1) if fmt else text, text)
    out=[]
    for r in re.finditer(r'<row>(.*?)</row>', xml, re.S):
        vals=[]
        for e in _elem.finditer(r.group(1)):
            tag,attrs,inner=e.group(1),e.group(2),e.group(3)
            ref=re.search(r'\bref="(\d+)"', attrs)
            if ref and ref.group(1) in ids:
                vals.append(ids[ref.group(1)])
            else:
                fmt=re.search(r'\bfmt="([^"]*)"', attrs); text=re.sub(r'<[^>]+>','',inner or '').strip()
                vals.append((tag, fmt.group(1) if fmt else text, text))
        out.append(vals)
    return out
