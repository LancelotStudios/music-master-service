# Verify analyze.py against SYNTHETIC audio with known ground truth (tempo/key/spectral).
# Run from master-service with the analysis deps installed:
#   PYTHONPATH=. python test_analyze.py
import numpy as np, soundfile as sf, tempfile, os, json
from analyze import analyze_wav
SR=44100; rng=np.random.default_rng(1)
def write(y,sr=SR):
    f=tempfile.mktemp(suffix=".wav"); sf.write(f,y.astype(np.float32),sr); return f
def env(n,a=0.002,d=0.12):
    t=np.arange(n)/SR; e=np.exp(-t/d); e[:int(a*SR)]*=np.linspace(0,1,int(a*SR)); return e
def kick():
    n=int(0.18*SR); t=np.arange(n)/SR; f=110*np.exp(-t/0.03)+50
    return np.sin(2*np.pi*np.cumsum(f)/SR)*env(n,d=0.12)
def snare():
    n=int(0.15*SR); return (rng.standard_normal(n)*0.7+np.sin(2*np.pi*180*np.arange(n)/SR)*0.3)*env(n,d=0.10)
def hat():
    n=int(0.05*SR); return rng.standard_normal(n)*env(n,d=0.03)*0.4
def groove(bpm,secs=20):
    y=np.zeros(int(secs*SR)); spb=60.0/bpm; eighth=spb/2; t=0.0; beat=0
    def add(s,at):
        i=int(at*SR); j=min(len(y),i+len(s)); y[i:j]+=s[:j-i]
    while t<secs-spb:
        add(kick(),t)                       # kick on every beat
        if beat%2==1: add(snare(),t)        # snare on 2 and 4
        add(hat(),t); add(hat(),t+eighth)   # hats on eighths
        t+=spb; beat+=1
    return 0.6*y/np.max(np.abs(y))
def chord_prog(seq,secs_each=0.8):
    y=[]
    for ch in seq:
        n=int(secs_each*SR); t=np.arange(n)/SR
        seg=sum(np.sin(2*np.pi*f*t)+0.3*np.sin(2*np.pi*2*f*t) for f in ch)/len(ch)
        seg*=(np.hanning(n)*0.4+0.6); y.append(seg)
    y=np.concatenate(y); return 0.5*y/np.max(np.abs(y))
C4,D4,E4,F4,G4,A4,B4,C5,D5=261.63,293.66,329.63,349.23,392.0,440.0,493.88,523.25,587.33
A3,Cn4,En4=220.0,261.63,329.63
res={}
for bpm in [120,90,75,140,100]:
    f=write(groove(bpm)); res[f"tempo_{bpm}"]=analyze_wav(f)["tempo"]; os.remove(f)
f=write(chord_prog([[C4,E4,G4],[F4,A4,C5],[G4,B4,D5],[C4,E4,G4]]*4)); r=analyze_wav(f); os.remove(f)
res["key_Cmajor"]=r["key"]; res["spectral_Cmaj"]=r["spectral"]
f=write(chord_prog([[A3,Cn4,En4],[D4,F4,A4],[E4,G4,B4],[A3,Cn4,En4]]*4)); res["key_Aminor"]=analyze_wav(f)["key"]; os.remove(f)
print(json.dumps(res,indent=2))
