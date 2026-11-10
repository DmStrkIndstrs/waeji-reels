import sys,subprocess,numpy as np
from faster_whisper import WhisperModel
m=WhisperModel('small',compute_type='int8')
for e in sys.argv[1:]:
    subprocess.run(['ffmpeg','-loglevel','error','-y','-i',f'/root/{e}/vo.mp3','-ac','1','-ar','16000',f'/root/{e}/vo16.wav'],check=True)
    import wave
    w=wave.open(f'/root/{e}/vo16.wav'); a=np.frombuffer(w.readframes(w.getnframes()),dtype=np.int16).astype(np.float32)/32768
    segs,_=m.transcribe(a,language='ko',word_timestamps=True)
    print('==',e,len(a)/16000)
    for s in segs: print(f'{s.start:.2f}-{s.end:.2f}',s.text)
