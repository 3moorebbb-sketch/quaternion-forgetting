// Linear-probe test: after Task B, is Task A still linearly decodable from the last hidden layer?
// Probes the same seed's network at three points: untrained init, Task A checkpoint, after 1000 SGD steps of Task B.
// Probe = logistic regression fit exactly (Newton/IRLS, ridge 1e-3) on 2000 random points, scored on the fixed 50×50 grid.
const QCL=require('./engine.js');
const an=process.argv[2],seed=+process.argv[3];const ar=QCL.arch(an);
const feats=(th,x1,x2)=>{const fw=ar.forward(th,x1,x2);return Array.from(fw.acts?fw.acts[fw.acts.length-2]:fw.h2);};
const GRID=[];for(let gy=0;gy<50;gy++)for(let gx=0;gx<50;gx++){const x1=gx/49*2-1,x2=1-gy/49*2;GRID.push([x1,x2,QCL.labelA(x1,x2)]);}
function solve(A,b){const n=b.length;for(let i=0;i<n;i++){let p=i;for(let r=i+1;r<n;r++)if(Math.abs(A[r][i])>Math.abs(A[p][i]))p=r;[A[i],A[p]]=[A[p],A[i]];[b[i],b[p]]=[b[p],b[i]];for(let r=i+1;r<n;r++){const f=A[r][i]/A[i][i];for(let c=i;c<n;c++)A[r][c]-=f*A[i][c];b[r]-=f*b[i];}}const x=new Array(n).fill(0);for(let i=n-1;i>=0;i--){let s=b[i];for(let c=i+1;c<n;c++)s-=A[i][c]*x[c];x[i]=s/A[i][i];}return x;}
function fitProbe(X,y){const D=X[0].length+1,w=new Array(D).fill(0);
 for(let it=0;it<30;it++){const H=Array.from({length:D},()=>new Array(D).fill(0)),g=new Array(D).fill(0);
  for(let i=0;i<X.length;i++){const xi=[...X[i],1];let z=0;for(let d=0;d<D;d++)z+=w[d]*xi[d];const p=1/(1+Math.exp(-z)),s=Math.max(p*(1-p),1e-9);
   for(let a=0;a<D;a++){g[a]+=(p-y[i])*xi[a];for(let c=0;c<D;c++)H[a][c]+=s*xi[a]*xi[c];}}
  for(let a=0;a<D;a++){g[a]+=1e-3*w[a];H[a][a]+=1e-3;}const step=solve(H,g);let mx=0;for(let d=0;d<D;d++){w[d]-=step[d];mx=Math.max(mx,Math.abs(step[d]));}if(mx<1e-8)break;}
 return w;}
function probeAcc(th){const r=QCL.makeRng(seed*31+7),X=[],y=[];for(let i=0;i<2000;i++){const x1=r()*2-1,x2=r()*2-1;X.push(feats(th,x1,x2));y.push(QCL.labelA(x1,x2));}
 const w=fitProbe(X,y);let c=0;for(const p of GRID){const f=[...feats(th,p[0],p[1]),1];let z=0;for(let d=0;d<f.length;d++)z+=w[d]*f[d];if((z>=0?1:0)===p[2])c++;}return 100*c/GRID.length;}
const init=ar.init(QCL.makeRng(seed*7919+1));                     // the same draw the checkpoint started from
const ck=QCL.trainTaskA(QCL.makeRng(seed*7919+1),ar);
const run=QCL.setup({arch:an,seed,lr:0.035,methods:['sgd']},ck);QCL.advance(run,1000);
const th=run.models.sgd,[headA,headB]=QCL.evaluate(th,ar);
// Frozen old readout: the checkpoint's own last layer on top of the drifted hidden layers
const hyb=new Float64Array(th);
if(ar.LAYOUT){const L=ar.LAYOUT[ar.LAYOUT.length-1],end=L.b+4*L.outS;for(let k=L.q;k<end;k++)hyb[k]=ck.th[k];}
else{const H=an==='real14'?14:24,W3=2*H+H+H*H+H;for(let k=W3;k<ar.P;k++)hyb[k]=ck.th[k];}
const oldHeadA=QCL.evaluate(hyb,ar)[0];
// Readout refit from the shared 100-point memory only (the same stored labelled inputs replay uses)
function memProbe(thx,lam){const mem=QCL.makeMemory(ck.th,QCL.makeRng(seed*7919+2),100,ar);const X=mem.map(p=>feats(thx,p[0],p[1])),y=mem.map(p=>p[2]);
 const D=X[0].length+1,w=new Array(D).fill(0);
 for(let it=0;it<50;it++){const Hm=Array.from({length:D},()=>new Array(D).fill(0)),g=new Array(D).fill(0);
  for(let i=0;i<X.length;i++){const xi=[...X[i],1];let z=0;for(let d=0;d<D;d++)z+=w[d]*xi[d];const p=1/(1+Math.exp(-z)),sv=Math.max(p*(1-p),1e-9);for(let a=0;a<D;a++){g[a]+=(p-y[i])*xi[a];for(let c=0;c<D;c++)Hm[a][c]+=sv*xi[a]*xi[c];}}
  for(let a=0;a<D;a++){g[a]+=lam*w[a];Hm[a][a]+=lam;}const stp=solve(Hm,g);let mx=0;for(let d=0;d<D;d++){w[d]-=stp[d];mx=Math.max(mx,Math.abs(stp[d]));}if(mx<1e-8)break;}
 let c=0;for(const p of GRID){const f=[...feats(thx,p[0],p[1]),1];let z=0;for(let d=0;d<f.length;d++)z+=w[d]*f[d];if((z>=0?1:0)===p[2])c++;}return 100*c/GRID.length;}
const memRefit=memProbe(th,0.1);
process.stdout.write(JSON.stringify({an,seed,ckpt:ck.acc,attempts:ck.attempts,headA,headB,probeInit:probeAcc(init),probeCkpt:probeAcc(ck.th),probeAfter:probeAcc(th),oldHeadA,memRefit})+'\n');
