// Task B with Adam instead of SGD. native: Adam on the network's own parameters (tangent-projected
// gradients + its own retraction, exactly as Task A is trained). wspace: convert an S³ checkpoint to
// w = A·q and run Euclidean Adam on w. Same checkpoint and Task B batches as the SGD study.
const Q=require('./engine.js');
const [seed,an,variant,lr]=[+process.argv[2],process.argv[3],process.argv[4],+process.argv[5]];
const src=Q.arch(an);const freeName={qs3:'qfree',qs3sep:'qfreesep'}[an];
function toFree(th){const FR=Q.arch(freeName),w=new Float64Array(FR.P);src.LAYOUT.forEach((L,l)=>{const F=FR.LAYOUT[l];for(let k=0;k<L.n;k++){const A=th[L.A+k];for(let c=0;c<4;c++)w[F.q+4*k+c]=A*th[L.q+4*k+c];}for(let c=0;c<4*L.outS;c++)w[F.b+c]=th[L.b+c];});return w;}
const ck=Q.trainTaskA(Q.makeRng(seed*7919+1),src);
const ar=variant==='wspace'?Q.arch(freeName):src;const th=variant==='wspace'?toFree(ck.th):new Float64Array(ck.th);
const P=ar.P,m=new Float64Array(P),v=new Float64Array(P),b1=0.9,b2=0.999;
const rng=Q.makeRng(seed*104729+3);let hit=null,hitStep=null,fin,B;
for(let s=1;s<=1000;s++){
 const batch=[];for(let i=0;i<20;i++){const x1=rng()*2-1,x2=rng()*2-1;batch.push([x1,x2,Q.labelB(x1,x2)]);}
 for(let j=0;j<20;j++)rng();
 const g=new Float64Array(P);
 for(const p of batch){const fw=ar.forward(th,p[0],p[1]);ar.backward(th,fw,'B',3.5*(Q.sigmoid(fw.zB)-p[2])/20,g);}
 ar.tangentProject(th,g);const st=new Float64Array(P);
 for(let i=0;i<P;i++){m[i]=b1*m[i]+(1-b1)*g[i];v[i]=b2*v[i]+(1-b2)*g[i]*g[i];st[i]=(m[i]/(1-b1**s))/(Math.sqrt(v[i]/(1-b2**s))+1e-8);}
 ar.applyStep(th,st,lr);
 if(s%10===0){const e=Q.evaluate(th,ar);if(hit===null&&e[1]>=95){hit=e[0];hitStep=s;}if(s===1000){fin=e[0];B=e[1];}}
}
process.stdout.write(JSON.stringify({seed,an,variant,lr,ck:ck.acc,hit,hitStep,fin,B})+'\n');
