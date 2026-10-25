const Q=require('./engine.js');const [an,seed]=[process.argv[2],+process.argv[3]];const ar=Q.arch(an);
const ck=Q.trainTaskA(Q.makeRng(seed*7919+1),ar);const out={an,seed,ck:ck.acc,runs:[]};
const LR=[0.0175,0.035,0.07];
const G={sgd:[{}],wanchor:[0.01,0.05,0.2,1].map(v=>({lambda:v})),oanchor:[0.01,0.03,0.1,0.3].map(v=>({mu:v})),
 replay:[0.25,0.5,1,2].map(v=>({rho:v})),togd:[5,15,40].map(v=>({K:v})),togdr:[{K:15,refresh:10}]};
for(const lr of LR)for(const[m,list]of Object.entries(G))for(const kv of list){
 const r=Q.runExperiment(Object.assign({arch:an,seed,lr,methods:[m],evalEvery:10},kv),ck);
 out.runs.push({m,lr,kv,h:r.history.map(x=>[x.step,+x[m][0].toFixed(2),+x[m][1].toFixed(2)])});}
require('fs').writeFileSync(`as/${an}_${seed}.json`,JSON.stringify(out));
