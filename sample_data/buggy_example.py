from os import *
from sys import *
import re, json, csv, math, random, hashlib, base64, struct

X = []
Y = []
Z = {}
W = {}
Q = None
R = []
CACHE = {}
tmp = []
data2 = {}

def p(x,y,z,w,q,r,s,t):
    a=[]
    b={}
    c=0
    d=[]
    if x>0:
        if y>0:
            if z>0:
                if w>0:
                    if q>0:
                        for i in range(x):
                            for j in range(y):
                                for k in range(z):
                                    if i%2==0:
                                        if j%3==0:
                                            if k%5==0:
                                                a.append(i*j*k)
                                                b[i]=[j,k,i+j+k]
                                                c+=1
                                                d.append({'i':i,'j':j,'k':k,'sum':i+j+k,'prod':i*j*k})
                                            elif k%7==0:
                                                a.append(i+j+k)
                                            else:
                                                pass
                                        elif j%11==0:
                                            for n in range(k):
                                                a.append(n)
                                    else:
                                        if j>5:
                                            c+=j
                    else:
                        c=-1
                else:
                    c=-2
            else:
                c=-3
        else:
            c=-4
    else:
        c=-5
    return a,b,c,d

def proc(lst):
    global X,Y,Z,W,Q,R,tmp
    res=[]
    for i in range(len(lst)):
        v=lst[i]
        if type(v)==int or type(v)==float:
            if v>100:
                if v<1000:
                    res.append(v*2)
                    X.append(v)
                    Z[v]=v*2
                else:
                    if v>5000:
                        if v<10000:
                            res.append(v/2)
                            Y.append(v)
                        else:
                            res.append(0)
                    else:
                        res.append(v)
                        W[v]=v
            elif v<0:
                if v>-50:
                    res.append(abs(v))
                    tmp.append(v)
                elif v<-1000:
                    res.append(v*-1)
                else:
                    res.append(0)
            else:
                res.append(v)
        elif type(v)==str:
            if len(v)>0:
                res.append(v.upper())
                Q=v
            else:
                res.append(None)
        elif type(v)==list:
            sub=proc(v)
            res.extend(sub)
        else:
            pass
    R=res
    return res

def do_stuff(fn,mode,enc,sep,skip,lim,col1,col2,col3,col4,col5):
    global CACHE,data2
    out=[]
    errs=[]
    cnt=0
    t=[]
    try:
        f=open(fn,mode,encoding=enc)
        lines=f.readlines()
        f.close()
        for i in range(len(lines)):
            if i<skip:
                continue
            if lim>0 and cnt>=lim:
                break
            ln=lines[i].strip()
            if len(ln)==0:
                continue
            parts=ln.split(sep)
            if len(parts)<5:
                errs.append(i)
                continue
            try:
                v1=int(parts[col1]) if col1<len(parts) else 0
                v2=float(parts[col2]) if col2<len(parts) else 0.0
                v3=str(parts[col3]) if col3<len(parts) else ''
                v4=parts[col4] if col4<len(parts) else None
                v5=parts[col5] if col5<len(parts) else None
                if v1>0 and v2>0.0 and len(v3)>0:
                    h=hashlib.md5((v3+str(v1)).encode()).hexdigest()
                    if h in CACHE:
                        out.append(CACHE[h])
                    else:
                        rec={'v1':v1,'v2':v2,'v3':v3,'v4':v4,'v5':v5,'hash':h,'idx':i}
                        CACHE[h]=rec
                        out.append(rec)
                        data2[h]={'idx':i,'v1':v1}
                    cnt+=1
                else:
                    errs.append(i)
            except:
                errs.append(i)
                continue
            t.append(cnt)
    except:
        pass
    return out,errs,cnt

def calc(n):
    if n==0: return 1
    if n==1: return 1
    if n<0: return -1
    a=0;b=1;c=0
    for i in range(2,n+1):
        c=a+b;a=b;b=c
    return c

def chk(s):
    if s==None: return False
    if len(s)==0: return False
    o=0;c=0
    for ch in s:
        if ch=='(': o+=1
        elif ch==')':
            c+=1
            if c>o: return False
        elif ch=='[': o+=1
        elif ch==']':
            c+=1
            if c>o: return False
    return o==c

myfunc = lambda x,y,z,w: (x+y)*z-w if x>0 else (y-x)*w+z if y>0 else x*y*z*w
myfunc2 = lambda a,b,c,d,e,f: a*b+c*d-e*f if all([a,b,c,d,e,f]) else 0
myfunc3 = lambda p,q: p**2+2*p*q+q**2
myfunc4 = lambda n: [i for i in range(n) if all(i%j!=0 for j in range(2,i)) and i>1]
myfunc5 = lambda x: x if x<2 else myfunc5(x-1)+myfunc5(x-2)
