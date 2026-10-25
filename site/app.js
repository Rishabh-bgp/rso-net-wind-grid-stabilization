fetch('model.json').then(r=>r.json()).then(M=>{
  const D=M.demo, js=D.windows.map(w=>WindModel.forecast(M,w));
  // chart
  const c=document.getElementById('chart'),g=c.getContext('2d'),W=c.width,H=c.height,p=36;
  const all=D.actual.concat(js,D.persistence),lo=Math.floor(Math.min(...all)),hi=Math.ceil(Math.max(...all));
  const X=i=>p+i*(W-2*p)/(D.actual.length-1),Y=v=>H-p-(v-lo)*(H-2*p)/(hi-lo||1);
  g.strokeStyle='rgba(18,18,18,.10)';g.fillStyle='#5a4a3a';g.font='12px sans-serif';
  for(let v=lo;v<=hi;v++){g.beginPath();g.moveTo(p,Y(v));g.lineTo(W-p,Y(v));g.stroke();g.fillText(v+' m/s',2,Y(v)+4);}
  const line=(a,col,w)=>{g.strokeStyle=col;g.lineWidth=w;g.beginPath();a.forEach((v,i)=>i?g.lineTo(X(i),Y(v)):g.moveTo(X(i),Y(v)));g.stroke();};
  line(D.persistence,'#c9a227',1.6);line(D.actual,'#121212',1.8);line(js,'#b3121b',2.2);
  // metrics
  const t=document.getElementById('mt');[['MAE','MAE (m/s)'],['RMSE','RMSE (m/s)'],['MAPE_%','MAPE (%)'],['R2','R²']].forEach(([k,n])=>{
    t.insertAdjacentHTML('beforeend',`<tr><td>${n}</td><td>${M.metrics.model[k].toFixed(3)}</td><td>${M.metrics.persistence[k].toFixed(3)}</td></tr>`);});
  // sliders
  const $=id=>document.getElementById(id);
  function upd(){const ws=+$('ws').value,wd=+$('wd').value,tr=+$('tr').value;$('wsv').textContent=ws.toFixed(1)+' m/s';$('wdv').textContent=wd+'°';$('trv').textContent=tr.toFixed(2);
    const win=[];for(let i=0;i<M.lookback;i++){const v=Math.max(0,ws-tr*(M.lookback-1-i));win.push(WindModel.features(v,wd));}
    if(M.kind==='rso'){const q=WindModel.forecast(M,win,true);$('fc').textContent=q[1].toFixed(2)+' m/s';$('band').textContent=`80% band (P10–P90): ${q[0].toFixed(2)} – ${q[2].toFixed(2)} m/s. Any grid action would be advisory only and need human approval.`;}
    else $('fc').textContent=WindModel.forecast(M,win).toFixed(2)+' m/s';
    $('ps').textContent=ws.toFixed(2)+' m/s';}
  ['ws','wd','tr'].forEach(i=>$(i).addEventListener('input',upd));upd();
});
