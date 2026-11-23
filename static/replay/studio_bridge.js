/* Adapter between the unchanged replay controls and the Studio screen router. */
(() => {
  const params = new URLSearchParams(location.search);
  const sourceKey = params.get('source') || 'lab';
  let sample = null;
  const mainSlot = document.createElement('div');
  mainSlot.id = 'studio-three-main';
  $('c3').hidden = true;
  $('c3').parentElement.prepend(mainSlot);
  const sensorSlot = document.createElement('div');
  sensorSlot.id = 'studio-three-camera';
  $('camA').hidden = true;
  $('cameraFeeds').append(sensorSlot);
  const threeStyle = document.createElement('style');
  threeStyle.textContent = `#c3,#camA{display:none!important}#studio-three-main{position:absolute;inset:0}#studio-three-main .scene-canvas{width:100%;height:100%;position:relative}#studio-three-camera{aspect-ratio:4/3;position:relative;border:1px solid var(--line);border-radius:6px;overflow:hidden}#studio-three-camera .scene-canvas{width:100%;height:100%;position:relative}#studio-three-camera canvas{height:100%!important;aspect-ratio:auto;border:0} .canvas-caption{position:absolute;bottom:10px;left:12px;font-size:11px;color:#d1d7df;pointer-events:none}.scene-loading{padding:20px;color:var(--muted)}.viewbar,#cameraFeeds{z-index:2}.base-axis-label{white-space:nowrap;font:10px system-ui;color:#e7edf4;background:#172129c9;border-radius:3px;padding:2px 4px;pointer-events:none}`;
  threeStyle.textContent += '.studio-focus3d .hd,.studio-focus3d .c-small,.studio-focus3d .col,.studio-focus3d .logcard{display:none}.studio-focus3d .main{display:block;height:100dvh;padding:8px;box-sizing:border-box}.studio-focus3d .c-3d{height:100%;display:flex;flex-direction:column}.studio-focus3d .cv{height:auto;flex:1;min-height:180px}';
  threeStyle.textContent += '.studio-focus3d #app{height:100dvh;padding:0;gap:0}.studio-focus3d .main{flex:1 1 auto}';
  document.head.append(threeStyle);
  // Studio owns source, episode, status and fullscreen controls. Keep transport and view controls here.
  const cleanStyle=document.createElement('style');
  cleanStyle.textContent='.hd,.c-3d>h2,#topFs{display:none!important}.c-3d{padding:0!important;overflow:hidden!important}.cv .viewbar{top:10px;left:10px;max-width:calc(100% - 20px)}.wrap{padding:10px;gap:10px}.canvas-caption{display:none}.studio-focus3d .main{padding:0}.studio-focus3d #app{padding:10px}.studio-focus3d .cv{min-height:200px}';
  cleanStyle.textContent+='#studio-three-camera .canvas-caption{display:block}';
  document.head.append(cleanStyle);
  const options=document.createElement('details');options.className='studio-view-options';
  const optionsLabel=document.createElement('summary');optionsLabel.textContent='Tùy chọn';options.append(optionsLabel);
  const optionsBody=document.createElement('div');options.append(optionsBody);
  optionsBody.append($('camTilt').closest('label'));
  const keysButton=document.querySelector('[popovertarget="keysPop"]');keysButton.textContent='Phím tắt 3D';optionsBody.append(keysButton);
  const legendButton=document.querySelector('.legend-btn');legendButton.textContent='Chú giải';optionsBody.append(legendButton);
  document.body.append($('legendPop'));
  document.querySelector('.viewbar').append(options);
  $('vCams').hidden=true;$('bCam').textContent='Camera cổ tay';
  cleanStyle.textContent+='.viewbar #vCams{display:none}.studio-view-options{position:relative;font-size:12px}.studio-view-options summary{cursor:pointer;padding:3px 9px;list-style:none}.studio-view-options>div{position:absolute;top:calc(100% + 10px);right:0;min-width:260px;padding:12px;border:1px solid var(--line);border-radius:8px;background:var(--card);box-shadow:0 8px 24px #0004;display:flex;flex-wrap:wrap;gap:8px}.studio-view-options:not([open])>div{display:none}.studio-view-options label.tilt{width:100%;padding:4px 0!important}.viewbar .sep{margin:3px 2px}';
  $('flt').value='all';$('mixResult').value='all';buildSel();
  $('c3').closest('.card').querySelector('h2 span').textContent = '3D · Three.js / WebGL';
  const note = document.createElement('p');
  note.className = 'note';
  note.textContent = 'Tọa độ hiển thị theo đế: XY tại trục khớp 1, Z = 0 trên mặt bàn. Điều kiện vật lý và ngưỡng Z của env dùng world MuJoCo.';
  $('info').before(note);
  const publish = (fr, fs) => {
    const stream=sourceKey+'/'+ep;
    if(!sample || sample.stream!==stream || sample.step!==i){
      const continuous=!!timer && sample?.stream===stream && i===sample.step+1;
      sample={stream,step:i,time:performance.timeOrigin+performance.now(),durationMs:STEP_DT*1000/speed,continuous};
    }
    const computed = fkPose(fr);
    const poses = Object.fromEntries(Object.entries(computed).map(([name,p]) => [name,{p:p.p,R:p.R.flat()}]));
    if(fr.pose) Object.assign(poses,fr.pose);
    if(CAM_TILT_EXTRA && computed.wrist_cam_arm) poses.wrist_cam_arm={p:computed.wrist_cam_arm.p,R:computed.wrist_cam_arm.R.flat()};
    const origin = [...(S.pick_region?.pan || [0,-.15]),.42];
    const sensor = S.cams?.arm_cam;
    let camera = null;
    if(sensor){
      const T = sensor.body && poses[sensor.body];
      const R = T ? [T.R.slice(0,3),T.R.slice(3,6),T.R.slice(6,9)] : null;
      camera = {p:T?mvec(R,sensor.pos).map((v,k)=>v+T.p[k]):sensor.pos,R:(T?mmul(R,sensor.R):sensor.R).flat(),fovy:sensor.fovy,mount:T?{bodyPose:T,local:{p:sensor.pos,R:sensor.R.flat()}}:undefined};
    }
    const lines = [];
    for(const [enabled,from,to,color] of [[OPT.past,pastStart(),i,'#63d7b1'],[OPT.future,i,futureEnd(),'#f4ae62']]){
      if(!enabled)continue;
      if(OPT.path && to>from)lines.push({points:fs.slice(from,to+1).filter((_,k)=>k%Math.max(1,Math.ceil((to-from)/150))===0||k===to-from).map(f=>f.ee),color,opacity:.65});
      ghostIdx(i,from===i?to:from).forEach(k=>lines.push({points:[...fs[k].arm,fs[k].ee],color,opacity:.18}));
    }
    parent.postMessage({type:'studio-replay-scene',source:sourceKey,scene:{poses,objects:replayObjects(fr).map((obj,k)=>({...obj,id:obj.id||`object-${k+1}`,x:obj.pos[0],y:obj.pos[1],yaw:obj.yaw||0})),bowl:S.bowl,bowlGeoms:S.bowl_geoms,ee:fr.ee,origin,region:S.pick_region,zones:S.pick_region?{region:S.pick_region,stop:S.unreachable_region||null,scan:S.pick_scan||[]}:null,showRegion:$('showReach').checked,sensor:SHOW_CAM?camera:null,camera:{...CAM,t:[...CAM.t]},target:fr.tg,lines,contacts:{box:fr.ct,table:fr.tb,bowl:fr.bc||[],cam:fr.cc||[]},playing:!!timer,sample:{...sample,continuous:!!timer&&sample.continuous}}},location.origin);
  };
  window.studioRender = publish;
  const notify = () => parent.postMessage({type:'studio-replay-state',source:sourceKey,episode:ep,step:i,playing:!!timer}, location.origin);
  window.studioPlaybackChanged=()=>{if(!timer)render();publish(F()[i],F());notify();};
  let stamp = '', controlled = false;
  const observe = () => {if(!controlled)return;const next=`${ep}/${i}/${!!timer}`;if(next!==stamp){stamp=next;notify();}};
  new MutationObserver(observe).observe(document.getElementById('stepCount'), {attributes:true,childList:true,subtree:true});
  document.addEventListener('click', () => queueMicrotask(observe));
  window.addEventListener('message', event => {
    if(event.origin!==location.origin||event.source!==parent||event.data?.type!=='studio-replay-command')return;
    const command=event.data;
    controlled=true;
    if(typeof command.focus3d==='boolean' && document.documentElement.classList.contains('studio-focus3d')!==command.focus3d){document.documentElement.classList.toggle('studio-focus3d',command.focus3d);render();}
    if(command.orbit){Object.assign(CAM,command.orbit);syncFollow();return;}
    if((command.theme==='light'||command.theme==='dark')&&document.documentElement.dataset.studioTheme!==command.theme){document.documentElement.dataset.studioTheme=command.theme;render();}
    if(Number.isInteger(command.episode)&&command.episode>=0&&command.episode<EPS.length&&command.episode!==ep){
      $('flt').value='all';$('mixResult').value='all';buildSel();loadEp(command.episode);
    }
    if(Number.isFinite(command.step)&&command.step!==i)go(command.step);
    if([.1,.25,.5,1,1.5,2,3,4].includes(Number(command.speed)) && Number(command.speed)!==speed){const wasPlaying=!!timer;if(wasPlaying)stop();speed=Number(command.speed);$('spd').value=String(speed);if($('speedValue'))$('speedValue').value=`${speed}×`;if(wasPlaying)play();}
    if(command.play===true&&!timer&&F().length>1)play();
    if(command.play===false&&timer)stop();
    observe();
    publish(F()[i],F());
  });
  const style=document.createElement('style');
  style.textContent='html[data-studio-theme="dark"]{color-scheme:dark;--bg:#14171c;--card:#1d2128;--ink:#e6e9ee;--muted:#9aa3af;--line:#343a44;--table:#4a4130}html[data-studio-theme="light"]{color-scheme:light;--bg:#f6f7f9;--card:#fff;--ink:#1c2128;--muted:#6b7280;--line:#d8dce2;--table:#e6d7b5}';
  document.head.append(style);
  parent.postMessage({type:'studio-replay-ready',source:sourceKey},location.origin);
  publish(F()[i],F());
  observe();
})();
