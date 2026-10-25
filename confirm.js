const Q=require('./engine.js');const [an,seed]=[process.argv[2],+process.argv[3]];const ar=Q.arch(an);
const ck=Q.trainTaskA(Q.makeRng(seed*7919+1),ar);const out={an,seed,ck:ck.acc,attempts:ck.attempts,r:{}};
for(const lr of [0.0175,0.035]){const r=Q.runExperiment({arch:an,seed,lr,methods:['sgd'],evalEvery:10},ck);const h=r.history.map(x=>[x.step,x.sgd[0],x.sgd[1]]);const hit=h.find(x=>x[2]>=95);out.r[lr]={hit:hit?hit[1]:null,step:hit?hit[0]:null,fin:h.at(-1)[1],B:h.at(-1)[2]};}
process.stdout.write(JSON.stringify(out)+'\n');
