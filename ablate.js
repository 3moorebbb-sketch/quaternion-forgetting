// Ablations of the S³ cost on the split-head network. Every variant starts from the SAME
// qs3sep Task A checkpoint and sees the SAME Task B batches; only the Task B update differs.
const Q=require('./engine.js');const S3=Q.arch('qs3sep'),FR=Q.arch('qfreesep');
const [seed,variant,lr]=[+process.argv[2],process.argv[3],+process.argv[4]];
const L3=S3.LAYOUT,LF=FR.LAYOUT;
function toFree(th){const w=new Float64Array(FR.P);L3.forEach((L,l)=>{const F=LF[l];for(let k=0;k<L.n;k++){const A=th[L.A+k];for(let c=0;c<4;c++)w[F.q+4*k+c]=A*th[L.q+4*k+c];}for(let c=0;c<4*L.outS;c++)w[F.b+c]=th[L.b+c];});return w;}
function step(ar,th,g,lr,v){
 if(v==='riem')return ar.applyStep(th,g,lr);                       // engine's update (exp map on S³, A floored)
 if(v==='wspace')return ar.applyStep(th,g,lr);                     // FR arch: Euclidean in w = A·q
 if(v==='euclid'){for(let i=0;i<th.length;i++)th[i]-=lr*g[i];L3.forEach(L=>{for(let k=0;k<L.n;k++)th[L.A+k]=Math.max(0.01,th[L.A+k]);});return;} // A·q, plain steps, no unit constraint
 // the remaining variants keep S³ but change the step
 S3.tangentProject(th,g);
 L3.forEach(L=>{for(let k=0;k<L.n;k++){const o=L.q+4*k,A=th[L.A+k];
   let vx=g[o],vy=g[o+1],vz=g[o+2],vw=g[o+3];
   if(v==='precond'){const s=1/(A*A);vx*=s;vy*=s;vz*=s;vw*=s;}       // undo the A² factor: induced Δw ≈ Euclidean Δw
   const nv=Math.hypot(vx,vy,vz,vw);
   if(nv>1e-12){const c=Math.cos(lr*nv),sn=Math.sin(lr*nv)/nv;let q0=c*th[o]-sn*vx,q1=c*th[o+1]-sn*vy,q2=c*th[o+2]-sn*vz,q3=c*th[o+3]-sn*vw;const nq=Math.hypot(q0,q1,q2,q3);th[o]=q0/nq;th[o+1]=q1/nq;th[o+2]=q2/nq;th[o+3]=q3/nq;}
   const An=A-lr*g[L.A+k];th[L.A+k]=v==='nofloor'?An:Math.max(0.01,An);}
  for(let c=L.b;c<L.b+4*L.outS;c++)th[c]-=lr*g[c];});
}
const ck=Q.trainTaskA(Q.makeRng(seed*7919+1),S3);
const ar=variant==='wspace'?FR:S3;let th=variant==='wspace'?toFree(ck.th):new Float64Array(ck.th);
const rng=Q.makeRng(seed*104729+3);let hit=null,hitStep=null;const h=[];
const ev=()=>Q.evaluate(th,ar);let e=ev();h.push([0,e[0],e[1]]);
for(let s=1;s<=1000;s++){
 const batch=[];for(let i=0;i<20;i++){const x1=rng()*2-1,x2=rng()*2-1;batch.push([x1,x2,Q.labelB(x1,x2)]);}
 for(let j=0;j<20;j++)rng();                                        // memory indices (consumed by the engine)
 const g=new Float64Array(ar.P);
 for(const p of batch){const fw=ar.forward(th,p[0],p[1]);ar.backward(th,fw,'B',1*3.5*(Q.sigmoid(fw.zB)-p[2])/20,g);}
 ar.tangentProject(th,g);step(ar,th,g,lr,variant);
 if(s%10===0){e=ev();h.push([s,e[0],e[1]]);if(hit===null&&e[1]>=95){hit=e[0];hitStep=s;}}
}
// mean |A| and share of direction vs magnitude change for the S³-parameterised variants
let dirShare=null;if(ar===S3){let dq=0,dA=0;L3.forEach(L=>{for(let k=0;k<L.n;k++){const o=L.q+4*k,A0=ck.th[L.A+k],A1=th[L.A+k];
 const w0=[0,1,2,3].map(c=>A0*ck.th[o+c]),w1=[0,1,2,3].map(c=>A1*th[o+c]);const mag=Math.abs(Math.abs(A1)*Math.hypot(...[0,1,2,3].map(c=>th[o+c]))-Math.abs(A0));
 const tot=Math.hypot(...w0.map((x,i)=>w1[i]-x));dA+=mag;dq+=tot;}});dirShare=+(1-dA/Math.max(dq,1e-12)).toFixed(3);}
process.stdout.write(JSON.stringify({seed,variant,lr,ck:ck.acc,hit,hitStep,fin:h.at(-1)[1],B:h.at(-1)[2],dirShare})+'\n');
