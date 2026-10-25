const fs=require('fs');const ARCH=['qs3','qs3sep','qfree','qfreesep','real14','real24'],M=['sgd','wanchor','oanchor','replay','togd','togdr'];
const mean=a=>a.reduce((p,c)=>p+c,0)/a.length,sd=a=>{const m=mean(a);return Math.sqrt(a.reduce((p,c)=>p+(c-m)**2,0)/Math.max(1,a.length-1))};
function load(an){const out=[];for(let s=1;s<=12;s++){const f=`as/${an}_${s}.json`;if(fs.existsSync(f))out.push(JSON.parse(fs.readFileSync(f)));}return out;}
const key=r=>`${r.lr}|${JSON.stringify(r.kv)}`;
function metrics(h){const last=h[h.length-1];const hit=h.find(x=>x[1+1]>=95);return{A:last[1],B:last[2],Ahit:hit?hit[1]:null,stepHit:hit?hit[0]:null};}
function summarize(files,m,k){const v=files.map(f=>metrics(f.runs.find(r=>r.m===m&&key(r)===k).h));
 const hits=v.filter(x=>x.Ahit!=null);return{n:v.length,A:mean(v.map(x=>x.A)),Asd:sd(v.map(x=>x.A)),Amin:Math.min(...v.map(x=>x.A)),B:mean(v.map(x=>x.B)),
  Ahit:hits.length?mean(hits.map(x=>x.Ahit)):null,hitFrac:hits.length/v.length,stepHit:hits.length?mean(hits.map(x=>x.stepHit)):null,vals:v.map(x=>x.A)};}
// bootstrap CI for difference of means
function bootDiff(a,b,rng=(()=>{let s=12345;return()=>(s=(s*1103515245+12345)%2147483648)/2147483648})()){const d=[];for(let i=0;i<4000;i++){const ma=mean(a.map(()=>a[Math.floor(rng()*a.length)])),mb=mean(b.map(()=>b[Math.floor(rng()*b.length)]));d.push(ma-mb);}d.sort((x,y)=>x-y);return[d[100],d[3899]];}
const res={};const sel={};
for(const an of ARCH){const F=load(an);if(F.length<12){console.log(an,'only',F.length,'seeds so far');}if(!F.length)continue;
 const tune=F.filter(f=>f.seed<=6),test=F.filter(f=>f.seed>6);res[an]={ck:F.map(f=>f.ck),methods:{}};
 for(const m of M){const keys=[...new Set(F[0].runs.filter(r=>r.m===m).map(key))];
  // selection on tuning seeds: best A@1000 with B@1000 >= 95; fallback best B
  let best=null;for(const k of keys){const s=summarize(tune,m,k);const ok=s.B>=95;if(!best||(ok&&!best.ok)||(ok===best.ok&&(ok?s.A>best.s.A:s.B>best.s.B)))best={k,s,ok};}
  const t=test.length?summarize(test,m,best.k):null, all=summarize(F,m,best.k);
  // SGD-style forgetting at matched progress uses all configs' default lr as well
  res[an].methods[m]={cfg:best.k,tuneOK:best.ok,test:t,all,grid:keys.map(k=>({k,...(({A,B,Ahit,hitFrac})=>({A,B,Ahit,hitFrac}))(summarize(F,m,k))}))};}}
fs.writeFileSync('arch_results.json',JSON.stringify(res));
const f1=x=>x==null?'  —  ':x.toFixed(1);
console.log('\nHeld-out seeds 7–12 (config chosen on seeds 1–6: best A@1000 s.t. B@1000 ≥ 95)');
console.log('method   '+ARCH.map(a=>a.padEnd(26)).join(''));
for(const m of M){let line=m.padEnd(9);for(const an of ARCH){const r=res[an]&&res[an].methods[m];if(!r||!r.test){line+=''.padEnd(26);continue;}line+=`${f1(r.test.A)}±${r.test.Asd.toFixed(1)} B${f1(r.test.B)}${r.tuneOK?'':'*'}`.padEnd(26);}console.log(line);}
console.log('\nChosen configs (lr|knob):');for(const m of M)console.log(m.padEnd(9)+ARCH.map(an=>(res[an]?res[an].methods[m].cfg:'').padEnd(26)).join(''));
console.log('\nForgetting at matched progress — SGD, Task A when Task B first reaches 95% (all 12 seeds, per lr):');
for(const an of ARCH){if(!res[an])continue;console.log(an.padEnd(7),res[an].methods.sgd.grid.map(g=>`lr ${JSON.parse('['+g.k.split('|')[0]+']')[0]}: A@B95 ${f1(g.Ahit)} (reached ${Math.round(g.hitFrac*100)}%) A@1000 ${f1(g.A)}`).join('  ·  '));}
console.log('\nDifference vs qs3 on held-out seeds, 95% bootstrap CI of mean A@1000:');
for(const m of M){let line=m.padEnd(9);for(const an of ARCH.slice(1)){const a=res[an]&&res[an].methods[m].test,b=res.qs3&&res.qs3.methods[m].test;if(!a||!b){line+=''.padEnd(28);continue;}const ci=bootDiff(a.vals,b.vals);line+=`${an}: ${(a.A-b.A>=0?'+':'')+(a.A-b.A).toFixed(1)} [${ci[0].toFixed(1)},${ci[1].toFixed(1)}]`.padEnd(30);}console.log(line);}
