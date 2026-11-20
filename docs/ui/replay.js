const PHASE_NAME = {
  0:"nhấc thẳng lên từ home", 1:"bay ngang tới phía TRÊN HỘP", 2:"hạ xuống pre-grasp", 3:"hạ xuống vị trí KẸP",
  4:"đóng kẹp, chờ 30 bước", 5:"nhấc hộp lên", 6:"bay ngang tới phía TRÊN BÁT", 7:"hạ hộp xuống bát",
  8:"mở kẹp thả hộp, chờ 20 bước", 9:"rút tay lên khỏi bát", 10:"về lại home", 11:"XONG"
};
const PHASE_COLOR = ["#6b7280","#2f6fd0","#0e9aa7","#d97706","#c026d3","#7c3aed","#2f6fd0","#0e9aa7","#c026d3","#6b7280","#6b7280","#1a9a3a"];
const WAIT_LIMIT = {4:30, 8:20};
const JSHORT = {shoulder_pan:"pan", shoulder_lift:"lift", elbow_flex:"elbow", wrist_flex:"wflex", wrist_roll:"wroll"};
// khớp k (0..4) quay đoạn tay nào: đoạn đó là link S.arm_bodies[k+1]
const JMEAN = "pan = xoay thân · lift = nâng vai · elbow = gập khuỷu · wflex = gập cổ tay · wroll = xoay cổ tay";

const app = document.getElementById("app");
if (!DATA?.episodes?.length) {
  app.innerHTML = '<div class="err"><h1>Chưa có dữ liệu</h1><p>Đây là file mẫu. Hãy chạy <code>python -m sim_engine.make_replay</code> để tạo <code>docs/replay.html</code>.</p></div>';
  throw new Error("no data");
}
const S = DATA.scene, EPS = DATA.episodes;
const f3 = v => v ? "(" + v.map(x => (x >= 0 ? "+" : "") + x.toFixed(3)).join(", ") + ")" : "—";
const dist = (a, b) => Math.hypot(a[0]-b[0], a[1]-b[1], a[2]-b[2]);
const cm = v => (v * 100).toFixed(1);

let ep = 0, i = 0, timer = null, speed = 1;   // speed = hệ số so với thời gian thực (1 = đúng tốc độ robot chạy thật)
const STEP_DT = S.step_dt || 1 / 30;           // giây mô phỏng mỗi bước (policy ra 1 action mỗi STEP_DT)
const OPT = { past:true, future:true, path:true, hz:"1", every:1 };

/* ---------- khung trang ---------- */
app.innerHTML = `
  <a href="#rng" class="skip">Đến thanh tua replay</a>
  <header class="hd">
    <h1 title="Mỗi lần sửa policy, chạy lại python -m sim_engine.make_replay để cập nhật file này">Replay</h1>
    <span class="seed-controls">
      <button id="sPrev" title="seed trước (phím [)">◀ seed</button>
      <label>Seed <select id="sel" title="chọn nhanh seed ở xa; dùng nút ◀ ▶ để duyệt lần lượt"></select></label>
      <button id="sNext" title="seed sau (phím ])">seed ▶</button>
      <span class="sub" id="epPos"></span>
    </span>
    <span class="seed-summary" id="res" role="status"></span>
    <span id="tags" class="sub"></span>
    <span class="badge" style="background:color-mix(in srgb,var(--ok) 20%,transparent);color:var(--ok)" title="Hộp chỉ được giữ bằng ma sát và va chạm thật">KẸP THẬT</span>
    <span class="sub">Policy <code>${S.policy}</code> · hình 2D vẽ lại từ số liệu thật, không phải vật lý</span>
    <span style="flex:1"></span>
    <button type="button" popovertarget="legendPop" class="legend-btn">Chú giải các góc nhìn</button>
    <div id="legendPop" popover class="legend-pop">
      <span><i style="background:var(--box)"></i>hộp</span><span><i style="background:var(--bowl)"></i>bát (nét đứt = ngưỡng vào bát)</span>
      <span><i style="background:var(--hand)"></i>tay (giữa 2 ngón)</span><span><i style="background:var(--goal)"></i>đích</span>
      <span><i style="background:var(--arm)"></i>Nét giữa = xương tay · vùng nhạt = bề rộng khâu · khung hộp = bộ kẹp. Đỏ = khâu chạm hộp, cam = chạm bàn hoặc sàn, <b style="color:#a855f7">tím = chạm bát</b>.</span>
      <span><i style="background:var(--past)"></i>cánh tay mờ xanh lục = QUÁ KHỨ</span><span><i style="background:var(--future)"></i>cánh tay mờ cam = TƯƠNG LAI</span>
      <span>5 khớp: ${JMEAN}. Đoạn <b>base</b> (đế) đứng yên; hàm kẹp (1 hàm cố định + 1 hàm di động) không có trong IK.</span>
      <span>số nhỏ cạnh cánh tay = lệch bao nhiêu bước so với hiện tại · chấm trên đường đứt = 1 bước</span>
    </div>
  </header>
  <main class="main">
    <div class="card c-small"><h2>NHÌN TỪ TRÊN XUỐNG (x ngang, y dọc, mét) · số 1-5 = khớp<label style="float:right;font-weight:400"><input type="checkbox" id="showReach" checked> vùng gắp được</label></h2><svg id="top" role="img" aria-label="Hình chiếu XY cố định" viewBox="0 0 520 340"></svg>
      <h2>Nhìn từ cạnh · Y ngang, Z cao (m)</h2><svg id="side" role="img" aria-label="Hình chiếu YZ cố định" viewBox="0 0 520 340"></svg></div>
    <div class="card c-3d"><h2><span>3D (VIEW CHÍNH)</span><span class="hint">kéo = xoay · lăn = zoom · nhấp đúp = về mặc định</span></h2>
      <div class="cv"><canvas id="c3"></canvas>
        <div class="viewbar" role="group" aria-label="Góc nhìn 3D">
          <button id="v1" title="Góc chéo mặc định">Chéo</button><button id="v2" title="Nhìn từ phía +Y về đế">Trước</button><button id="vSide" title="Chiếu vuông góc từ +X lên mặt phẳng YZ">Hông</button><button id="v3" title="Nhìn thẳng từ trên xuống">Trên</button><button id="vCams" title="Xem vị trí camera bên hông cổ tay">Vị trí cam</button>
          <span class="sep"></span><button id="bCam" aria-pressed="true" title="Bật hoặc tắt ảnh camera cổ tay">Ảnh cam</button><label class="tilt" title="Gập giá đỡ camera quanh chốt để đổi góc nhìn; giá trị ưng ý thì ghi vào config.WRIST_CAM_TILT_DEG">Gập cam <input type="range" id="camTilt" min="-45" max="45" step="1" value="0"><output id="camTiltV">0°</output></label>
        </div>
        <div id="cameraFeeds"><canvas id="camA" width="320" height="240" aria-label="Ảnh dựng lại từ camera cổ tay"></canvas></div>
      </div>
      <div class="ctl">
        <div class="transport mb-1.5 flex flex-wrap items-center gap-x-3 gap-y-2">
          <button id="bFirst" class="min-h-0 inline-flex h-7 w-7 items-center justify-center rounded-full border-0 bg-transparent !p-0 shrink-0 text-[var(--muted)] transition-colors hover:bg-[color-mix(in_srgb,var(--ink)_10%,transparent)] hover:text-[var(--ink)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--ink)]" title="Về bước đầu (Home)" aria-label="Về bước đầu"><svg class="h-[15px] w-[15px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 19V5"/><path d="m13 6-6 6 6 6"/><path d="M7 12h14"/></svg></button>
          <div class="inline-flex items-center gap-0.5 rounded-full bg-[color-mix(in_srgb,var(--ink)_7%,transparent)] px-1 py-0.5" role="group" aria-label="Điều khiển phát">
            <button id="bPP" class="min-h-0 inline-flex h-7 w-7 items-center justify-center rounded-full border-0 bg-transparent !p-0 shrink-0 text-[var(--muted)] transition-colors hover:bg-[color-mix(in_srgb,var(--ink)_10%,transparent)] hover:text-[var(--ink)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--ink)]" title="Phase trước (↑)" aria-label="Phase trước"><svg class="h-[15px] w-[15px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="19 20 9 12 19 4 19 20"/><line x1="5" x2="5" y1="19" y2="5"/></svg></button>
            <button id="bPrev" class="min-h-0 inline-flex h-7 w-7 items-center justify-center rounded-full border-0 bg-transparent !p-0 shrink-0 text-[var(--muted)] transition-colors hover:bg-[color-mix(in_srgb,var(--ink)_10%,transparent)] hover:text-[var(--ink)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--ink)]" title="Lùi 1 bước (←)" aria-label="Lùi 1 bước"><svg class="h-[15px] w-[15px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m15 18-6-6 6-6"/></svg></button>
            <button id="bPlay" class="min-h-0 inline-flex h-8 w-8 items-center justify-center rounded-full border-0 bg-[var(--ink)] !p-0 shrink-0 text-[var(--card)] transition-opacity hover:opacity-80 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--ink)]" title="Phát / dừng (Space)" aria-label="Phát"><svg class="h-[14px] w-[14px]" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="6 3 20 12 6 21 6 3"/></svg></button>
            <button id="bNext" class="min-h-0 inline-flex h-7 w-7 items-center justify-center rounded-full border-0 bg-transparent !p-0 shrink-0 text-[var(--muted)] transition-colors hover:bg-[color-mix(in_srgb,var(--ink)_10%,transparent)] hover:text-[var(--ink)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--ink)]" title="Tiến 1 bước (→)" aria-label="Tiến 1 bước"><svg class="h-[15px] w-[15px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m9 18 6-6-6-6"/></svg></button>
            <button id="bPhase" class="min-h-0 inline-flex h-7 w-7 items-center justify-center rounded-full border-0 bg-transparent !p-0 shrink-0 text-[var(--muted)] transition-colors hover:bg-[color-mix(in_srgb,var(--ink)_10%,transparent)] hover:text-[var(--ink)] focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--ink)]" title="Phase kế (↓)" aria-label="Phase kế"><svg class="h-[15px] w-[15px]" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="5 4 15 12 5 20 5 4"/><line x1="19" x2="19" y1="5" y2="19"/></svg></button>
          </div>
          <label class="ml-auto inline-flex items-center gap-2 text-xs text-[var(--muted)]">Tốc độ <select id="spd" class="min-h-0 h-7 rounded-full border border-solid border-[var(--line)] bg-transparent !px-2.5 !py-0 text-xs text-[var(--ink)]"><option value="0.5">chậm 0,5×</option><option value="1" selected>bình thường (thực tế)</option><option value="1.5">nhanh 1,5×</option><option value="2">nhanh 2×</option></select></label>
        </div>
        <div class="scrub">
          <label for="rng">Thời gian <output id="stepCount" for="rng"></output></label>
          <input type="range" id="rng" min="0" value="0">
        </div>
        <div class="phasebar" id="phasebar">
          <div class="tlcur" id="tlc"><button type="button" id="tlh" role="slider" aria-label="Kéo để chọn bước trong các phase" aria-valuemin="0" aria-valuenow="0"><svg viewBox="0 0 18 24" aria-hidden="true"><path class="pin" d="M3 1h12a2 2 0 0 1 1.5 3.3L10.4 17a2 2 0 0 1-2.8 0L1.5 4.3A2 2 0 0 1 3 1z"/><rect class="needle" x="8.2" y="16" width="1.6" height="8"/></svg></button></div>
          <div class="tl" id="tl"></div>
        </div>
        <details class="opts"><summary>Hiển thị quá khứ / tương lai</summary>
          <div class="row">
            <label><input type="checkbox" id="oPast" checked><span style="color:var(--past);font-weight:700">quá khứ</span></label>
            <label><input type="checkbox" id="oFut" checked><span style="color:var(--future);font-weight:700">tương lai</span></label>
            <label><input type="checkbox" id="oPath" checked>đường đi của tay</label>
            <label>phạm vi <select id="oHz"><option value="1" selected>1 bước</option><option value="5">5 bước</option><option value="15">15 bước</option><option value="phase">hết phase</option><option value="end">hết ván</option></select></label>
            <label>1 cánh tay mờ mỗi <select id="oEv"><option value="1" selected>1 bước</option><option value="3">3 bước</option><option value="5">5 bước</option><option value="10">10 bước</option></select></label>
          </div>
        </details>
        <span class="sub keys">Phím: ← → từng bước · ↑ ↓ phase · Space · Home/End</span>
      </div>
    </div>
    <div class="col">
      <div class="card"><h2>ĐÍCH · IK</h2><div id="info"></div></div>
      <div class="card"><h2>KẸP VÀ VÀO BÁT: điều kiện env kiểm tra</h2><div id="chk"></div></div>
    </div>
  </main>
<div class="card logcard"><h2>LOG TỪNG BƯỚC (bấm một dòng để nhảy tới)</h2><div class="logwrap" id="logw"></div></div>
`;
const $ = id => document.getElementById(id);
const epLabel = (e, k) => { const ex = (e.tags || []).filter(x => !/^(THÀNH CÔNG|HỎNG)/.test(x)); return `<option value="${k}">${k + 1}. ${e.name || "seed " + e.seed} — ${e.success ? "OK" : "FAIL"}${ex.length ? " · " + ex.join(" · ") : ""}</option>`; };
{ // dropdown chia nhóm (nếu có): chỉ dùng để nhảy tới seed ở xa; duyệt lần lượt bằng nút ◀ ▶
  const groups = [...new Set(EPS.map(e => e.group || ""))];
  $("sel").innerHTML = groups.map(g => g ? `<optgroup label="${g}">${EPS.map((e, k) => (e.group || "") === g ? epLabel(e, k) : "").join("")}</optgroup>`
    : EPS.map((e, k) => (e.group || "") === g ? epLabel(e, k) : "").join("")).join("");
}

/* ---------- ánh xạ tọa độ -> pixel ---------- */
const W = 520, H = 340;
const TOP = { x0:-0.42, x1:0.42, y0:-0.30, y1:0.24 };
const SIDE = { x0:-0.42, x1:0.42, z0:0.32, z1:0.89 };
// One pixel-per-metre scale for both axes within each orthographic view.
const scaleX = R => Math.min((W - 20) / (R.x1 - R.x0), (H - 20) / ((R.y1 ?? R.z1) - (R.y0 ?? R.z0)));
const sx = (v, R) => W / 2 + (v - (R.x0 + R.x1) / 2) * scaleX(R);
const sy = (v, R, low, high) => H / 2 - (v - (low + high) / 2) * scaleX(R);
const topX = x => sx(x, TOP);
const topY = y => sy(y, TOP, TOP.y0, TOP.y1);
const sideX = y => sx(y, SIDE);
const sideY = z => sy(z, SIDE, SIDE.z0, SIDE.z1);
const polyPts = (arr, fx, fy, a, b) => arr.map(p => fx(p[a]).toFixed(1) + "," + fy(p[b]).toFixed(1)).join(" ");
function cross(x, y, r, col, w) { return `<path d="M${x-r} ${y} H${x+r} M${x} ${y-r} V${y+r}" stroke="${col}" stroke-width="${w}"/>`; }
function F() { return EPS[ep].frames; }

/* ---------- quá khứ và tương lai ---------- */
function futureEnd() {
  const fs = F(), n = fs.length - 1, nx = Math.min(i + 1, n);
  if (OPT.hz === "end") return n;
  if (OPT.hz === "phase") { const cur = fs[nx].ph; let k = nx; while (k < n && fs[k+1].ph === cur) k++; return k; }
  return Math.min(n, i + (+OPT.hz));
}
/* ---------- cánh tay THẬT (SO-101): động học thuận từ góc khớp + lưới tam giác giản lược của từng link ---------- */
const mmul = (A, B) => [0,1,2].map(r => [0,1,2].map(c => A[r][0]*B[0][c] + A[r][1]*B[1][c] + A[r][2]*B[2][c]));
const mvec = (A, v) => [0,1,2].map(r => A[r][0]*v[0] + A[r][1]*v[1] + A[r][2]*v[2]);
function quatMat(q) {
  const [w, x, y, z] = q;
  return [[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]];
}
const KIN = (S.kin || []).map(k => ({ ...k, Rl: quatMat(k.quat) }));
let CAM_TILT_EXTRA = 0;                                 // độ gập thêm (rad) do thanh "Gập cam", cộng vào độ gập có sẵn trong model
function fkPose(fr) {                                    // tư thế world của từng link: base (đứng yên) -> ... -> hàm kẹp di động
  if (fr._pose) return fr._pose;
  const ang = {}; S.joints.forEach((n, k) => ang[n] = fr.jq[k] * Math.PI / 180); ang.gripper = fr.fo;
  const out = {};
  for (const k of KIN) {
    let R = k.Rl, p = k.pos;
    if (k.parent) { const P = out[k.parent]; R = mmul(P.R, k.Rl); p = mvec(P.R, k.pos).map((v, j) => v + P.p[j]); }
    if (k.name === "wrist_cam_arm" && CAM_TILT_EXTRA) { const c = Math.cos(CAM_TILT_EXTRA), s3 = Math.sin(CAM_TILT_EXTRA); R = mmul(R, [[1, 0, 0], [0, c, -s3], [0, s3, c]]); }   // gập thêm quanh trục x của chốt
    if (k.joint) { const c = Math.cos(ang[k.joint]), s2 = Math.sin(ang[k.joint]); R = mmul(R, [[c, -s2, 0], [s2, c, 0], [0, 0, 1]]); }
    out[k.name] = { R, p };
  }
  return (fr._pose = out);
}
function armWorld(fr) {                                  // đỉnh lưới của từng link trong toạ độ world
  if (fr._w) return fr._w;
  const pose = fkPose(fr), out = {};
  for (const nm in S.arm_mesh) {
    const { R, p } = pose[nm], v = S.arm_mesh[nm].v, w = new Float64Array(v.length);
    for (let i = 0; i < v.length; i += 3) {
      w[i] = R[0][0]*v[i] + R[0][1]*v[i+1] + R[0][2]*v[i+2] + p[0];
      w[i+1] = R[1][0]*v[i] + R[1][1]*v[i+1] + R[1][2]*v[i+2] + p[1];
      w[i+2] = R[2][0]*v[i] + R[2][1]*v[i+1] + R[2][2]*v[i+2] + p[2];
    }
    out[nm] = w;
  }
  return (fr._w = out);
}
let SM = p => p;                                        // ánh xạ điểm world -> toạ độ vẽ (mặc định giữ nguyên; hình cạnh dùng Y ngang)
const ARMCOL = "#e8b923";
function linkColor(fr, nm, main) {                       // đỏ = chạm hộp, cam = chạm bàn, tím = chạm bát
  if (!main) return ARMCOL;
  return (fr.ct || []).includes(nm) ? "#e5484d" : (fr.tb || []).includes(nm) ? "#f59e0b" : touchBowl(fr, nm) ? BOWLC : ARMCOL;
}
function meshColors(name, frame) {
  const highlight = linkColor(frame, name, true), hl = highlight !== ARMCOL ? hexRgbCam(highlight) : null;
  const groups = S.arm_mesh[name].colors || [];
  let group = 0, end = groups[0]?.[0] || 0;
  return face => {
    while (group+1 < groups.length && face >= end) end += groups[++group][0];
    const g = groups[group];
    return (hl && !g?.[2]) ? hl : (g?.[1] || hexRgbCam(ARMCOL));   // mesh camera (g[2] = 1) luôn giữ màu thật
  };
}
function armHulls(fr, fx, fy, a, b) {              // bóng 2D của từng link = bao lồi của các đỉnh đã chiếu
  let s = "";
  const W3 = armWorld(fr);
  for (const nm of Object.keys(S.arm_mesh)) {
    const w = W3[nm], pts = [];
    for (let i = 0; i < w.length; i += 6) {
      const q = SM([w[i], w[i+1], w[i+2]]);
      pts.push([fx(q[a]), fy(q[b])]);
    }
    const poly = P => P.map(p => p[0].toFixed(1) + "," + p[1].toFixed(1)).join(" ");
    s += `<polygon points="${poly(hull2(pts))}" fill="${linkColor(fr, nm, true)}" fill-opacity=".3" stroke="#3a2c00" stroke-opacity=".5" stroke-width=".8"/>`;
  }
  return s;
}
const BOWLC = "#a855f7";                              // màu nhãn "đang chạm bát"
const touchBowl = (fr, nm) => (fr.bc || []).includes(nm);
function wallCorners(g) {                             // 4 góc của tấm tường nhìn từ trên (x, y)
  const [a, c] = g.size, co = Math.cos(g.yaw), si = Math.sin(g.yaw);
  return [[-a,-c],[a,-c],[a,c],[-a,c]].map(([lx, ly]) => [g.pos[0] + lx*co - ly*si, g.pos[1] + lx*si + ly*co]);
}
const EDGES3 = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
function wallCorners3(g) {                            // 8 đỉnh (x, y, z) của tấm tường
  const [a, c, hz] = g.size, co = Math.cos(g.yaw), si = Math.sin(g.yaw);
  return Array.from({ length: 8 }, (_, k) => {
    const lx = (k & 1 ? 1 : -1) * a, ly = (k & 2 ? 1 : -1) * c, lz = (k & 4 ? 1 : -1) * hz;
    return [g.pos[0] + lx*co - ly*si, g.pos[1] + lx*si + ly*co, g.pos[2] + lz];
  });
}
function armSeg(fr, fx, fy, a, b, w, op, red, col) {
  const A = fr.arm.map(p => { const q = SM(p); return [q[a], q[b]]; }); { const q = SM(fr.ee); A.push([q[a], q[b]]); }
  let s = red ? armHulls(fr, fx, fy, a, b) : "";
  for (let k = 0; k < A.length - 1; k++) {
    const nm = S.arm_bodies[k];                       // đoạn k nằm giữa khớp k và khớp k+1, mang tên link nm
    const hit = red && fr.ct.includes(nm), tb = red && fr.tb.includes(nm), bw = red && touchBowl(fr, nm);
    const c = hit ? "var(--bad)" : tb ? "var(--warn)" : bw ? BOWLC : (col || "var(--arm)");
    s += `<line x1="${fx(A[k][0])}" y1="${fy(A[k][1])}" x2="${fx(A[k+1][0])}" y2="${fy(A[k+1][1])}" stroke="${c}" stroke-width="${red ? 2.2 : w}" stroke-linecap="round" opacity="${hit || tb || bw ? 1 : red ? .9 : op}"/>`;
  }
  if (red) for (let k = 1; k <= 5; k++)                // đánh số khớp 1..5 trên cánh tay chính
    s += `<circle cx="${fx(A[k][0])}" cy="${fy(A[k][1])}" r="7" fill="var(--card)" stroke="var(--ink)" stroke-width="1.2"/><text x="${fx(A[k][0])}" y="${fy(A[k][1]) + 3.2}" font-size="9" font-weight="700" text-anchor="middle" fill="var(--ink)">${k}</text>`;
  return s;
}
function pastStart() {
  const fs = F();
  if (OPT.hz === "end") return 0;
  if (OPT.hz === "phase") { const cur = fs[i].ph; let k = i; while (k > 0 && fs[k-1].ph === cur) k--; return k; }
  return Math.max(0, i - (+OPT.hz));
}
// Chọn các khung để vẽ cánh tay mờ: cách nhau OPT.every bước, tối đa 12 cánh tay mỗi phía
function ghostIdx(from, to) {
  const span = Math.abs(to - from); if (span === 0) return [];
  let g = OPT.every; if (span / g > 12) g = Math.ceil(span / 12);
  const out = [], dir = to > from ? 1 : -1;
  for (let k = g; k <= span; k += g) out.push(from + dir * k);
  if (out[out.length - 1] !== to) out.push(to);
  return out;
}
function trails(fx, fy, a, b, pass) {   // pass 'ghost' = cánh tay mờ (vẽ dưới cánh tay hiện tại), 'path' = đường đi (vẽ TRÊN cánh tay)
  const fs = F(), n = fs.length - 1; let s = "";
  const G = pass === "ghost", P = pass === "path";
  const lab = (f, off, col) => `<text x="${fx(SM(f.ee)[a]) + 5}" y="${fy(SM(f.ee)[b]) - 5}" font-size="9" fill="${col}" opacity=".9">${off > 0 ? "+" : ""}${off}</text>`;
  if (OPT.past && i > 0) {
    const st = pastStart(), idx = ghostIdx(i, st);
    if (OPT.path && P) {
      const past = fs.slice(st, i + 1);
      s += `<polyline points="${polyPts(past.map(f => SM(f.ee)), fx, fy, a, b)}" fill="none" stroke="var(--past)" stroke-width="3" opacity=".95"/>`;
      if (dist(past[0].box, past[past.length-1].box) > 0.003)
        s += `<polyline points="${polyPts(past.map(f => SM(f.box)), fx, fy, a, b)}" fill="none" stroke="var(--box)" stroke-width="2.5" opacity=".85"/>`;
    }
    if (G) idx.slice().reverse().forEach(k => {                       // vẽ cái xa nhất trước, cái gần đè lên
      const t = Math.abs(k - i) / Math.max(1, Math.abs(st - i));
      s += armSeg(fs[k], fx, fy, a, b, 5, .75 - .35 * t, false, "var(--past)") + lab(fs[k], k - i, "var(--past)");
    });
  }
  if (OPT.future && i < n) {
    const e = futureEnd(), idx = ghostIdx(i, e);
    if (OPT.path && P) {
      const seg = fs.slice(i, e + 1);
      if (dist(seg[0].box, seg[seg.length-1].box) > 0.003)
        s += `<polyline points="${polyPts(seg.map(f => SM(f.box)), fx, fy, a, b)}" fill="none" stroke="var(--box)" stroke-width="3" stroke-dasharray="5 3"/>`;
      s += `<polyline points="${polyPts(seg.map(f => SM(f.ee)), fx, fy, a, b)}" fill="none" stroke="var(--future)" stroke-width="3.2" stroke-dasharray="6 3"/>`;
      seg.slice(1).forEach(f => { s += `<circle cx="${fx(SM(f.ee)[a])}" cy="${fy(SM(f.ee)[b])}" r="3" fill="var(--future)"/>`; });
    }
    if (G) idx.slice().reverse().forEach(k => {
      const t = Math.abs(k - i) / Math.max(1, Math.abs(e - i));
      s += armSeg(fs[k], fx, fy, a, b, 5, .8 - .35 * t, false, "var(--future)") + lab(fs[k], k - i, "var(--future)");
    });
  }
  return s;
}

// Lưới + trục có số (mét). ystep: bước vạch dọc, ylab: bước ghi số.
function axes(R, fx, fy, vlo, vhi, ystep, ylab, xname, yname) {
  const num = v => (Math.abs(v) < 1e-9 ? 0 : v).toFixed(2);
  let s = "";
  for (let v = Math.ceil(R.x0 / 0.05 - 1e-9) * 0.05; v <= R.x1 + 1e-9; v += 0.05) {
    const X = fx(v), z0 = Math.abs(v) < 1e-9, major = Math.abs(v / 0.1 - Math.round(v / 0.1)) < 1e-6;
    s += `<line x1="${X}" y1="10" x2="${X}" y2="${H-10}" stroke="var(--muted)" stroke-width="${z0 ? 1.4 : .5}" opacity="${z0 ? .7 : major ? .35 : .18}"/>`;
    if (major) s += `<text x="${X}" y="${H-13}" font-size="9" text-anchor="middle" fill="var(--muted)">${num(v)}</text>`;
  }
  for (let v = Math.ceil(vlo / ystep - 1e-9) * ystep; v <= vhi + 1e-9; v += ystep) {
    const Y = fy(v), z0 = Math.abs(v) < 1e-9, lab = Math.abs(v / ylab - Math.round(v / ylab)) < 1e-6;
    s += `<line x1="10" y1="${Y}" x2="${W-10}" y2="${Y}" stroke="var(--muted)" stroke-width="${z0 ? 1.4 : .5}" opacity="${z0 ? .7 : lab ? .35 : .18}"/>`;
    if (lab && Y < H - 24) s += `<text x="13" y="${Y - 2}" font-size="9" fill="var(--muted)">${num(v)}</text>`;
  }
  s += `<text x="${W-12}" y="${H-26}" font-size="11" font-weight="700" text-anchor="end" fill="var(--ink)">${xname} →</text>`;
  s += `<text x="13" y="22" font-size="11" font-weight="700" fill="var(--ink)">↑ ${yname}</text>`;
  return s;
}
/* ---------- hộp có hướng (quaternion) để vẽ được cảnh hộp bị nghiêng/lật ---------- */
const bq = fr => fr.bq || [1, 0, 0, 0];
function qrot(q, v) {                                  // xoay vector v bằng quaternion q = [w,x,y,z]
  const [w, x, y, z] = q, tx = 2*(y*v[2] - z*v[1]), ty = 2*(z*v[0] - x*v[2]), tz = 2*(x*v[1] - y*v[0]);
  return [v[0] + w*tx + (y*tz - z*ty), v[1] + w*ty + (z*tx - x*tz), v[2] + w*tz + (x*ty - y*tx)];
}
function boxV(c, q, hs) {                              // 8 đỉnh hộp (cùng thứ tự với cube cũ)
  const V = [];
  for (const sx of [-1,1]) for (const sy of [-1,1]) for (const sz of [-1,1]) { const r = qrot(q, [sx*hs, sy*hs, sz*hs]); V.push([c[0]+r[0], c[1]+r[1], c[2]+r[2]]); }
  return V;
}
function tiltDeg(fr) {                                 // góc giữa trục z của hộp và phương thẳng đứng
  const q = bq(fr), cz = Math.max(-1, Math.min(1, 1 - 2*(q[1]*q[1] + q[2]*q[2])));
  return Math.acos(Math.abs(cz)) * 180 / Math.PI;
}
function hull2(pts) {                                  // bao lồi (monotone chain) của các điểm 2D
  const P = pts.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]); if (P.length < 3) return P;
  const cr = (o, a, b) => (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0]);
  const lo = [], up = [];
  for (const p of P) { while (lo.length >= 2 && cr(lo[lo.length-2], lo[lo.length-1], p) <= 0) lo.pop(); lo.push(p); }
  for (const p of P.slice().reverse()) { while (up.length >= 2 && cr(up[up.length-2], up[up.length-1], p) <= 0) up.pop(); up.push(p); }
  return lo.slice(0, -1).concat(up.slice(0, -1));
}
const boxPoly = (fr, fx, fy, a, b) => hull2(boxV(fr.box, bq(fr), S.box_half).map(v => [fx(v[a]), fy(v[b])])).map(p => p.join(",")).join(" ");
/* ---------- vùng vươn tới của cánh tay (hình vành khuyên quanh trục khớp 1) ---------- */
const BANDC = ["#0e9aa7", "#d97706", "#8b8f98"];
const reachOn = () => S.reach && $("showReach") && $("showReach").checked;
const cpath = (cx, cy, r) => `M${cx + r},${cy}a${r},${r} 0 1 0 ${-2*r},0a${r},${r} 0 1 0 ${2*r},0`;
function reachTop(k) {                         // MỘT vùng màu nhạt: nơi cánh tay gắp được hộp (cắt theo mặt bàn)
  if (!reachOn()) return "";
  const R = S.reach, cx = topX(R.base[0]), cy = topY(R.base[1]), h = R.r_step / 2;
  const tx0 = topX(-S.table[0]), tx1 = topX(S.table[0]), ty0 = topY(S.table[1]), ty1 = topY(-S.table[1]);
  let s = `<clipPath id="tblclip"><rect x="${tx0}" y="${ty0}" width="${tx1 - tx0}" height="${ty1 - ty0}"/></clipPath><g clip-path="url(#tblclip)">`;
  R.bands[0].free.forEach(([a, r]) => { s += `<path d="${cpath(cx, cy, (r + h) * k)}${cpath(cx, cy, Math.max(0, a - h) * k)}" fill="#0e9aa7" fill-opacity=".17" fill-rule="evenodd"/>`; });
  return s + `</g>`;
}
function reachInfo(fs) {
  if (!S.reach) return "";
  const R = S.reach, b = fs[0].box, w = S.bowl, bx = R.base[0], by = R.base[1];
  const rb = Math.hypot(b[0] - bx, b[1] - by), rw = Math.hypot(w[0] - bx, w[1] - by);
  const inn = (iv, r) => iv.some(([lo, hi]) => r >= lo - R.r_step / 2 && r <= hi + R.r_step / 2);
  const fmt = iv => iv.length ? iv.map(([lo, hi]) => (lo * 100).toFixed(0) + "–" + (hi * 100).toFixed(0)).join(", ") + " cm" : "không";
  const mk = ok => `<span class="${ok ? "y" : "n"}">${ok ? "✓" : "✗"}</span>`;
  const g = R.bands[0], p = R.bands[1];
  let h = `<div class="note" style="margin:8px 0 3px"><b>Vùng cánh tay gắp được</b> (vùng màu nhạt trên bàn; tính từ trục khớp 1)</div>`;
  h += `<div class="chk"><span>${mk(inn(g.free, rb))}</span><span>hộp cách trục <b>${cm(rb)}</b> cm; gắp được ở <span class="mono">${fmt(g.free)}</span>, kẹp thẳng đứng ở <span class="mono">${fmt(g.vertical)}</span> ${mk(inn(g.vertical, rb))}</span></div>`;
  h += `<div class="chk"><span>${mk(inn(p.free, rw))}</span><span>bát cách trục <b>${cm(rw)}</b> cm; thả được ở <span class="mono">${fmt(p.free)}</span>, thả thẳng đứng ở <span class="mono">${fmt(p.vertical)}</span> ${mk(inn(p.vertical, rw))}</span></div>`;
  return h;
}
function drawTop(fr, fs) {
  const k = scaleX(TOP), b = S.bowl, b0 = fs[0].box, h = S.box_half * k;
  let s = `<rect x="${topX(-S.table[0])}" y="${topY(S.table[1])}" width="${topX(S.table[0])-topX(-S.table[0])}" height="${topY(-S.table[1])-topY(S.table[1])}" fill="var(--table)" opacity=".5"/>`;
  s += axes(TOP, topX, topY, TOP.y0, TOP.y1, 0.05, 0.1, "x (m)", "y (m)");
  s += reachTop(k);
  s += `<text x="${topX(b[0])}" y="${topY(b[1]) - 0.06*k}" font-size="10" text-anchor="middle" fill="var(--bowl)" font-weight="700">bát</text>`;
  S.bowl_geoms.forEach(g => {                           // bát thật: đáy tròn + 24 tấm tường hẹp xếp thành vòng tròn (gần đúng hình trụ rỗng)
    if (g.type === "cylinder") s += `<circle cx="${topX(g.pos[0])}" cy="${topY(g.pos[1])}" r="${g.size[0]*k}" fill="var(--bowl)" opacity=".2"/>`;
    else s += `<polygon points="${wallCorners(g).map(p => topX(p[0]) + "," + topY(p[1])).join(" ")}" fill="var(--bowl)" fill-opacity=".4" stroke="var(--bowl)" stroke-width="1"/>`;
  });
  s += `<circle cx="${topX(b[0])}" cy="${topY(b[1])}" r="${S.bowl_radius*k}" fill="none" stroke="var(--bowl)" stroke-dasharray="4 3"/>`;
  s += `<rect x="${topX(b0[0])-h}" y="${topY(b0[1])-h}" width="${2*h}" height="${2*h}" fill="none" stroke="var(--box)" stroke-dasharray="3 3" opacity=".6"/>`;
  s += `<text x="${topX(b0[0])}" y="${topY(b0[1]) + h + 11}" font-size="9" text-anchor="middle" fill="var(--box)">hộp ban đầu</text>`;
  if (Math.hypot(fr.box[0]-b0[0], fr.box[1]-b0[1]) > 0.004)     // mũi tên: hộp đã lệch khỏi chỗ cũ
    s += `<line x1="${topX(b0[0])}" y1="${topY(b0[1])}" x2="${topX(fr.box[0])}" y2="${topY(fr.box[1])}" stroke="var(--bad)" stroke-width="2" stroke-dasharray="2 2"/>`;
  s += trails(topX, topY, 0, 1, "ghost");
  s += armSeg(fr, topX, topY, 0, 1, 7, .85, true);
  s += trails(topX, topY, 0, 1, "path");
  s += `<polygon points="${boxPoly(fr, topX, topY, 0, 1)}" fill="var(--box)" stroke="#00000066"/>`;
  if (fr.tg) {
    s += `<line x1="${topX(fr.ee[0])}" y1="${topY(fr.ee[1])}" x2="${topX(fr.tg[0])}" y2="${topY(fr.tg[1])}" stroke="var(--warn)" stroke-width="2"/>`;
    s += `<circle cx="${topX(fr.tg[0])}" cy="${topY(fr.tg[1])}" r="7" fill="none" stroke="var(--goal)" stroke-width="2.5"/>`;
  }
  s += cross(topX(fr.ee[0]), topY(fr.ee[1]), 8, "var(--hand)", 2.5);
  $("top").innerHTML = s;
}
function drawSide(fr, fs) {
  const b = S.bowl;
  SM = ([x, y, z]) => [y, x, z];
  const bm = SM(b);
  const hullPoly = (pts, fx, fy, a, c) => hull2(pts.map(q => { const m = SM(q); return [fx(m[a]), fy(m[c])]; })).map(p => p.join(",")).join(" ");
  let s = `<rect x="10" y="${sideY(0.42)}" width="${W-20}" height="${H-10-sideY(0.42)}" fill="var(--table)" opacity=".5"/>`;
  s += axes(SIDE, sideX, sideY, SIDE.z0, SIDE.z1, 0.05, 0.05, "y (m)", "z (m)");
  s += `<line x1="10" y1="${sideY(0.42)}" x2="${W-10}" y2="${sideY(0.42)}" stroke="var(--muted)" stroke-width="2"/>`;
  s += `<text x="${W-12}" y="${sideY(0.42) + 12}" font-size="10" text-anchor="end" fill="var(--muted)">mặt bàn z=0.42</text>`;
  for (const g of S.bowl_geoms) {
    if (g.type === "cylinder") {
      const [, y, z] = g.pos, [r, halfHeight] = g.size;
      s += `<rect x="${sideX(y-r)}" y="${sideY(z+halfHeight)}" width="${sideX(y+r)-sideX(y-r)}" height="${sideY(z-halfHeight)-sideY(z+halfHeight)}" fill="var(--bowl)" opacity=".25"/>`;
    } else {
      s += `<polygon points="${hullPoly(wallCorners3(g), sideX, sideY, 0, 2)}" fill="var(--bowl)" fill-opacity=".18" stroke="var(--bowl)" stroke-width="1"/>`;
    }
  }
  s += `<text x="${sideX(bm[0])}" y="${sideY(b[2]+S.bowl_height)-5}" text-anchor="middle" fill="var(--bowl)" font-size="10">bát</text>`;
  const rim = b[2] + S.bowl_height, lim = rim - S.box_half;
  s += `<line x1="${sideX(bm[0]-0.09)}" y1="${sideY(rim)}" x2="${sideX(bm[0]+0.09)}" y2="${sideY(rim)}" stroke="var(--bowl)" stroke-width="1.2"/>`;
  s += `<text x="${sideX(bm[0]+0.095)}" y="${sideY(rim) + 3}" font-size="9" fill="var(--bowl)">miệng bát ${rim.toFixed(2)}</text>`;
  s += `<line x1="${sideX(bm[0]-0.09)}" y1="${sideY(lim)}" x2="${sideX(bm[0]+0.09)}" y2="${sideY(lim)}" stroke="var(--bowl)" stroke-width="1" stroke-dasharray="2 3"/>`;
  // hộp ban đầu (nét đứt) và hộp hiện tại
  const b0 = fs[0].box, B0 = SM(b0);
  s += `<polygon points="${hull2(boxV(b0, bq(fs[0]), S.box_half).map(v => { const m = SM(v); return [sideX(m[0]), sideY(m[2])]; })).map(p => p.join(",")).join(" ")}" fill="none" stroke="var(--box)" stroke-dasharray="3 3" opacity=".6"/>`;
  if (Math.hypot(fr.box[0]-b0[0], fr.box[1]-b0[1]) > 0.004)
    s += `<line x1="${sideX(B0[0])}" y1="${sideY(B0[2])}" x2="${sideX(SM(fr.box)[0])}" y2="${sideY(fr.box[2])}" stroke="var(--bad)" stroke-width="2" stroke-dasharray="2 2"/>`;
  s += trails(sideX, sideY, 0, 2, "ghost");
  s += armSeg(fr, sideX, sideY, 0, 2, 7, .85, true);
  s += trails(sideX, sideY, 0, 2, "path");
  s += `<polygon points="${hullPoly(boxV(fr.box, bq(fr), S.box_half), sideX, sideY, 0, 2)}" fill="var(--box)" stroke="#00000066"/>`;
  if (fr.tg) {
    const t = SM(fr.tg), e = SM(fr.ee);
    s += `<line x1="${sideX(e[0])}" y1="${sideY(e[2])}" x2="${sideX(t[0])}" y2="${sideY(t[2])}" stroke="var(--warn)" stroke-width="2"/>`;
    s += `<circle cx="${sideX(t[0])}" cy="${sideY(t[2])}" r="7" fill="none" stroke="var(--goal)" stroke-width="2.5"/>`;
  }
  { const e = SM(fr.ee); s += cross(sideX(e[0]), sideY(e[2]), 8, "var(--hand)", 2.5); }
  SM = p => p;
  $("side").innerHTML = s;
}

/* ---------- thông tin và cảnh báo ---------- */
function reasonOf(fr) {
  if (fr.phn === fr.ph) return "";
  if (fr.tg) return dist(fr.ee, fr.tg) < fr.th ? "tới đích" : "TIMEOUT 200 bước (không tới được đích)";
  return "hết thời gian chờ";
}
function drawInfo(fr, fs) {
  const ph = fr.ph, d = fr.tg ? dist(fr.ee, fr.tg) : null, lim = WAIT_LIMIT[ph] || 200, wait = !!WAIT_LIMIT[ph];
  const prev = fs[Math.max(0, i - 1)], box0 = fs[0].box;
  let al = "";

  // Cảnh báo 1: IK tin tới được nhưng tay thật không tới -> tay có thể bị chặn
  if (d !== null && fr.ie !== null && fr.ie < 0.005 && d > fr.th && fr.ps >= 5)
    al += `<div class="alert bad"><b>IK tin là tới được</b> (sai số ${cm(fr.ie)} cm) <b>nhưng tay thật còn cách đích ${cm(d)} cm</b> (ngưỡng ${cm(fr.th)}). Tay có thể bị cản.${fr.tb.length ? " Đang chạm bàn: <b>" + fr.tb.join(", ") + "</b>." : ""}</div>`;
  // Cảnh báo 2: sắp timeout
  if (!wait && fr.ps >= 0.75 * lim) al += `<div class="alert warn">Phase đã chạy <b>${fr.ps}/${lim}</b> bước: sắp timeout, cả ván có ${S.max_steps} bước.</div>`;
  // Cảnh báo 3: hộp vừa bị đẩy
  const mv = dist(fr.box, prev.box);
  if (!fr.gr && mv > 0.002) al += `<div class="alert bad"><b>Hộp vừa bị đẩy ${cm(mv)} cm</b> trong bước này${fr.ct.length ? " (chạm: <b>" + fr.ct.join(", ") + "</b>)" : ""}.</div>`;
  // Cảnh báo 4: hộp đã lệch so với ban đầu trước khi kẹp
  const off = Math.hypot(fr.box[0]-box0[0], fr.box[1]-box0[1]);
  if (!fr.gr && ph <= 4 && off > 0.01) al += `<div class="alert warn">Hộp đã lệch <b>${cm(off)} cm</b> so với vị trí ban đầu, trước khi được kẹp.</div>`;
  // Cảnh báo: hộp bị nghiêng/lật trước khi kẹp
  if (!fr.gr && tiltDeg(fr) > 3) al += `<div class="alert bad"><b>Hộp đang bị nghiêng ${tiltDeg(fr).toFixed(0)}°</b> (bị lật/gạt)${fr.ct.length ? ", đang chạm: <b>" + fr.ct.join(", ") + "</b>" : ""}.</div>`;
  // Cảnh báo 5: tay đang chạm bát
  if ((fr.bc || []).length) al += `<div class="alert warn"><b>Tay đang chạm bát</b>: <b style="color:${BOWLC}">${fr.bc.join(", ")}</b>. Bát có thể cản tay hoặc bị đẩy lệch.</div>`;
  const r = reasonOf(fr);
  if (r) al += `<div class="alert info">Bước này kết thúc phase ${ph} (<b>${r}</b>) → phase ${fr.phn}: ${PHASE_NAME[fr.phn] || ""}</div>`;

  // Bảng toạ độ: mỗi dòng một vật, cột x y z thẳng hàng để đối chiếu
  const row = (name, c, v) => `<tr style="color:${c}"><td>${name}</td>${v ? v.map(x => `<td class="num">${x.toFixed(3)}</td>`).join("") : '<td class="num">—</td><td class="num">—</td><td class="num">—</td>'}</tr>`;
  const dv3 = fr.tg ? fr.ee.map((x, k) => x - fr.tg[k]) : null;
  const tl = tiltDeg(fr), tilted = tl > 3;
  let html = al + `<div class="sec">ĐÍCH <em>tay đang nhắm tới đâu (mét)</em></div>
    <table class="ct"><thead><tr><th></th><th>x</th><th>y</th><th>z</th></tr></thead><tbody>
      ${row("Tay", "var(--hand)", fr.ee)}${row("Đích", "var(--goal)", fr.tg)}${row("Tay − đích", "var(--muted)", dv3)}${row("Hộp", "var(--box)", fr.box)}${row("Bát", "var(--bowl)", S.bowl)}
    </tbody></table>
    <div class="note" style="margin-top:2px">Bát: đáy z=${S.bowl[2].toFixed(2)}, miệng z=${(S.bowl[2]+S.bowl_height).toFixed(2)}, lòng rộng ±${(S.bowl_radius*100).toFixed(1)} cm. Hộp nghiêng <span class="${tilted ? "hlb" : ""}">${tl.toFixed(1)}°</span> so với phương thẳng đứng${tilted ? " (đang bị nghiêng/lật)" : ""}.</div>`;
  if (d !== null) {
    const pct = Math.min(100, d / 0.2 * 100), tp = Math.min(100, fr.th / 0.2 * 100), ok = d < fr.th;
    html += `<div class="meter" title="vạch đen = ngưỡng" style="margin-top:6px"><i style="width:${pct}%;background:${ok ? "var(--ok)" : "var(--warn)"}"></i><u style="left:${tp}%"></u></div>
      <div>Tay cách đích <span class="big ${ok ? "hlg" : "hlw"}">${cm(d)} cm</span> · ngưỡng ${cm(fr.th)} cm ${ok ? '<span class="hlg">đủ gần</span>' : '<span class="hlw">chưa đủ</span>'}</div>`;
  } else html += `<div class="note" style="margin-top:5px">Phase này đứng yên, chỉ đóng/mở kẹp và chờ, nên không có đích.</div>`;

  // IK: góc khớp IK muốn so với góc thật, và đoạn tay chạm hộp/bàn/bát
  html += `<div class="sec">IK <em>${fr.ik ? "tay dự đoán cách đích <span class=\"" + (fr.ie < 0.005 ? "hlg" : "hlw") + "\">" + cm(fr.ie) + " cm</span> (chưa tính va chạm)" : "phase chờ: IK không chạy"}</em></div>`;
  html += `<table class="jt"><thead><tr><th>#</th><th>khớp</th><th class="r">IK muốn</th><th class="r">thật</th><th class="r">lệch</th><th>đoạn tay</th></tr></thead><tbody>` +
    S.joints.map((jn, k) => {
      const link = S.arm_bodies[k + 1], a = fr.ik ? fr.ik[k] : null, dv = a === null ? null : a - fr.jq[k], ad = Math.abs(dv), cls = dv === null ? "" : ad > 15 ? "hlb" : ad > 5 ? "hlw" : "";
      const st = (fr.ct.includes(link) ? ' <span class="badge bad">chạm HỘP</span>' : "") +
        (fr.tb.includes(link) ? ' <span class="badge" style="background:color-mix(in srgb,var(--warn) 20%,transparent);color:var(--warn)">chạm bàn</span>' : "") +
        (touchBowl(fr, link) ? ` <span class="badge" style="background:color-mix(in srgb,${BOWLC} 20%,transparent);color:${BOWLC}">chạm BÁT</span>` : "");
      return `<tr><td><b>${k + 1}</b></td><td title="${jn}">${JSHORT[jn] || jn}</td><td class="r mono">${a === null ? "—" : a.toFixed(1)}</td><td class="r mono">${fr.jq[k].toFixed(1)}</td><td class="r mono ${cls}">${dv === null ? "—" : (dv >= 0 ? "+" : "") + dv.toFixed(1)}</td><td class="mono">${link.replace("_link", "")}${st}</td></tr>`;
    }).join("") + `</tbody></table>`;
  $("info").innerHTML = html;
}
// Từ lúc env đã "dính" hộp vào tay ít nhất một lần, hộp dịch chuyển là chủ ý
function ph_after_grasp() { return F().slice(0, i + 1).some(f => f.gr); }
// PUSH[ep] = chỉ số khung đầu tiên mà hộp trượt ngang > 0.2 cm khi chưa được kẹp (hoặc null)
const PUSH = EPS.map(e => {
  const fs = e.frames;
  for (let k = 1; k < fs.length; k++) {
    if (fs[k].gr) return null;
    if (Math.hypot(fs[k].box[0] - fs[k-1].box[0], fs[k].box[1] - fs[k-1].box[1]) > 0.002) return k;
  }
  return null;
});
function yn(ok, text) { return `<div class="chk"><span class="${ok ? "y" : "n"}">${ok ? "✓" : "✗"}</span><span>${text}</span></div>`; }
function drawChk(fr) {
  const dEB = dist(fr.ee, fr.box), z = fr.box[2], bz = S.bowl[2];
  const rim = bz + S.bowl_height, zMax = rim - S.box_half;
  const zOk = z > bz && z < zMax, hOk = fr.hd < S.bowl_radius, relOk = !fr.gr, setOk = (fr.bs ?? 0) < S.settled_speed;
  const clsy = ok => ok ? "hlg" : "hlb";
  const fixedOn = (fr.ct || []).includes("gripper"), movOn = (fr.ct || []).includes("moving_jaw_so101_v1");
  let h = `<div class="note" style="margin:0 0 3px"><b>Kẹp thật</b> (chỉ ma sát + va chạm, 1 hàm cố định + 1 hàm di động):</div>`;
  h += yn(fr.ag < -0.5, `lệnh đóng kẹp: <span class="${clsy(fr.ag < -0.5)}">${fr.ag}</span>`);
  h += yn(fixedOn, `hàm cố định chạm hộp`);
  h += yn(movOn, `hàm di động chạm hộp (góc ${fr.fo.toFixed(2)} rad)`);
  h += `<div class="chk"><span>→</span><span>Hai hàm cùng chạm hộp (đang kẹp): <span class="big ${clsy(fr.gr)}">${fr.gr ? "CÓ" : "KHÔNG"}</span></span></div>`;
  h += `<div class="note" style="margin:8px 0 3px"><b>Hộp vào bát</b> khi đủ cả 4 điều (env._get_info):</div>`;
  h += yn(hOk, `tâm hộp cách tâm bát <span class="${clsy(hOk)}">${cm(fr.hd)} cm</span> (cần &lt; ${(S.bowl_radius*100).toFixed(1)} = lòng bát)`);
  h += yn(zOk, `tâm hộp cao <span class="${clsy(zOk)}">${z.toFixed(3)}</span> trong (${bz.toFixed(2)}, ${zMax.toFixed(3)}): đỉnh hộp thấp hơn miệng bát ${rim.toFixed(2)}`);
  h += yn(relOk, `đã thả hộp (không còn bị kẹp giữ)`);
  h += yn(setOk, `hộp nằm yên: <span class="${clsy(setOk)}">${(fr.bs ?? 0).toFixed(3)}</span> m/s (cần &lt; ${S.settled_speed})`);
  h += `<div class="chk"><span>→</span><span>Thành công: <span class="big ${clsy(fr.ok)}">${fr.ok ? "CÓ" : "chưa"}</span></span></div>`;
  h += `<div class="note" style="margin:8px 0 3px"><b>Vật đang chạm hộp</b> (trừ mặt bàn):</div>`;
  h += fr.ct.length ? fr.ct.map(n => `<span class="badge ${n === "forearm_link" ? "bad" : "ok"}">${n}</span> `).join("") : '<span class="note">không có</span>';
  h += `<div class="note" style="margin:8px 0 3px"><b>Khúc tay đang chạm bàn:</b></div>`;
  h += fr.tb.length ? fr.tb.map(n => `<span class="badge bad">${n}</span> `).join("") : '<span class="note">không có</span>';
  h += `<div class="note" style="margin:8px 0 3px"><b>Khúc tay / ngón đang chạm bát:</b></div>`;
  h += (fr.bc || []).length ? fr.bc.map(n => `<span class="badge" style="background:color-mix(in srgb,${BOWLC} 20%,transparent);color:${BOWLC}">${n}</span> `).join("") : '<span class="note">không có</span>';

  // Hộp có bị lệch khỏi chỗ ban đầu không?
  const b0 = F()[0].box, dx = fr.box[0] - b0[0], dy = fr.box[1] - b0[1], dz = fr.box[2] - b0[2], off = Math.hypot(dx, dy);
  const carried = fr.gr || ph_after_grasp(fr);
  const oc = carried ? "" : off > 0.01 ? "hlb" : off > 0.005 ? "hlw" : "hlg";
  h += `<div class="note" style="margin:8px 0 3px"><b>Hộp có bị lệch không?</b> (so với bước 0)</div>`;
  h += `<div class="kv mono"><b>ban đầu</b><span>${f3(b0)}</span><b>hiện tại</b><span>${f3(fr.box)}</span><b>lệch ngang</b><span class="${oc}">${cm(off)} cm</span> <b>(dx, dy, dz)</b><span>${(dx*100).toFixed(1)}, ${(dy*100).toFixed(1)}, ${(dz*100).toFixed(1)} cm</span></div>`;
  h += carried ? `<div class="note">Hộp đã được kẹp/mang đi nên dịch chuyển là chủ ý.</div>`
    : off > 0.01 ? `<div class="alert bad" style="margin-top:3px">Hộp bị đẩy lệch <b>${cm(off)} cm</b> khi chưa được kẹp: hướng đích sẽ lệch theo.</div>` : `<div class="note">Hộp gần như đứng yên (lệch &lt; 0.5 cm là bình thường).</div>`;
  const pu = PUSH[ep];
  h += pu ? `<div class="note" style="margin-top:3px">Bị đẩy lần đầu ở <a href="#" id="jmpPush"><b>bước ${F()[pu].t}</b> (phase ${F()[pu].ph})</a> bởi <b>${F()[pu].ct.join(", ") || "?"}</b>.</div>` : `<div class="note" style="margin-top:3px">Không bị đẩy trước khi kẹp.</div>`;
  h += reachInfo(F());
  $("chk").innerHTML = h;
  const jp = $("jmpPush"); if (jp) jp.onclick = e => { e.preventDefault(); go(pu); };
}

/* ---------- log ---------- */
function maxDiff(f) { return f.ik ? Math.max(...f.ik.map((a, k) => Math.abs(a - f.jq[k]))) : null; }
function buildLog(fs) {
  let rows = `<table><thead><tr><th>bước</th><th>phase</th><th>tay</th><th>đích</th><th>cách đích</th><th>IK sai số</th><th>góc lệch max</th><th>hộp</th><th>ngón</th><th>dính</th><th>chạm hộp</th><th>chạm bàn</th></tr></thead><tbody>`;
  fs.forEach((f, k) => {
    const d = f.tg ? dist(f.ee, f.tg) : null, md = maxDiff(f);
    const dCls = d === null ? "" : d < f.th ? "hlg" : "hlw";
    const iCls = f.ie === null ? "" : f.ie < 0.005 ? "" : "hlw";
    const mCls = md === null ? "" : md > 15 ? "hlb" : md > 5 ? "hlw" : "";
    const moved = k > 0 && !f.gr && dist(f.box, fs[k-1].box) > 0.002;
    rows += `<tr data-k="${k}" class="${f.ph !== f.phn ? "chg" : ""}"><td><button type="button" class="step-link" aria-label="Đến bước ${f.t}">${f.t}</button></td><td>${f.ph}${f.ph !== f.phn ? " → " + f.phn : ""}</td><td class="mono">${f3(f.ee)}</td><td class="mono">${f3(f.tg)}</td><td class="${dCls}">${d === null ? "—" : cm(d) + " cm"}</td><td class="${iCls}">${f.ie === null ? "—" : cm(f.ie) + " cm"}</td><td class="${mCls}">${md === null ? "—" : md.toFixed(1) + "°"}</td><td class="mono ${moved ? "hlb" : ""}">${f3(f.box)}</td><td>${f.fo.toFixed(3)}</td><td>${f.gr ? "có" : "không"}</td><td class="${f.ct.length ? "hlb" : ""}">${f.ct.join(", ")}</td><td>${f.tb.join(", ")}</td></tr>`;
  });
  $("logw").innerHTML = rows + "</tbody></table>";
  $("logw").querySelectorAll("tr[data-k]").forEach(tr => tr.onclick = () => go(+tr.dataset.k));
}
let SEGS = [];
function buildTimeline(fs) {
  const n = fs.length - 1; SEGS = []; let start = 1;
  for (let k = 1; k <= n; k++) if (k === n || fs[k].ph !== fs[k+1].ph) { SEGS.push([fs[k].ph, start, k]); start = k + 1; }
  // chiều rộng tỉ lệ với số bước của phase (phase nào lâu nhất thấy ngay), nhưng tối thiểu 26px để luôn bấm được;
  // bấm một đoạn = nhảy tới bước đầu của phase đó
  $("tl").innerHTML = SEGS.map(([p, a, b]) => `<div data-a="${a}" title="phase ${p}: ${PHASE_NAME[p] || ""} (bước ${a}-${b}, ${b-a+1} bước)" style="--w:${b-a+1};background:${PHASE_COLOR[p]}">${p}</div>`).join("");
  // (kéo thả được xử lý ở bộ lắng nghe của #phasebar, bên dưới)
}
function tlPos(i) {                                    // vị trí vạch con trỏ (%) theo đúng bề rộng thật của các đoạn
  if (!SEGS.length || i <= 0) return 0;
  const k = Math.max(0, SEGS.findIndex(([, a, b]) => i >= a && i <= b)), [, a, b] = SEGS[k];
  const el = $("tl").children[k], W = $("tl").clientWidth;
  return W ? (el.offsetLeft + (i - a + 0.5) / (b - a + 1) * el.offsetWidth) / W * 100 : 0;
}
function stepAtX(clientX) {                           // ngược lại của tlPos: vị trí chuột trên thanh phase -> bước
  const tl = $("tl"), x = clientX - tl.getBoundingClientRect().left, els = [...tl.children];
  if (!els.length) return 0;
  let k = els.findIndex(el => x < el.offsetLeft + el.offsetWidth + 1);
  if (k < 0) k = els.length - 1;
  const el = els[k], [, a, b] = SEGS[k], f = Math.min(1, Math.max(0, (x - el.offsetLeft) / el.offsetWidth));
  return Math.min(b, a + Math.floor(f * (b - a + 1)));
}
{
  const bar = $("phasebar"); let down = false;
  bar.addEventListener("pointerdown", e => { down = true; try { bar.setPointerCapture(e.pointerId); } catch (err) {} stop(); go(stepAtX(e.clientX)); e.preventDefault(); });
  bar.addEventListener("pointermove", e => { if (down) go(stepAtX(e.clientX)); });
  const end = e => { down = false; try { bar.releasePointerCapture(e.pointerId); } catch (err) {} };
  bar.addEventListener("pointerup", end); bar.addEventListener("pointercancel", end);
}

/* ---------- 3D: tự vẽ bằng canvas 2D, không cần thư viện ---------- */
const CAM0 = { yaw:-0.95, pitch:0.5, dist:0.95, orthographic:false };
const CAM = { ...CAM0, t:[0.02, -0.03, 0.5] };
const JN = ["pan", "lift", "elbow", "wflex", "wroll"];
const css = n => getComputedStyle(document.documentElement).getPropertyValue("--" + n).trim();
function makeProj(w, h) {
  const cp = Math.cos(CAM.pitch), sp = Math.sin(CAM.pitch), cy = Math.cos(CAM.yaw), sy = Math.sin(CAM.yaw);
  const cam = [CAM.t[0] + CAM.dist*cp*cy, CAM.t[1] + CAM.dist*cp*sy, CAM.t[2] + CAM.dist*sp];
  const f = [-cp*cy, -cp*sy, -sp];
  let r = [f[1], -f[0], 0]; const rl = Math.hypot(r[0], r[1]) || 1; r = [r[0]/rl, r[1]/rl, 0];
  const u = [r[1]*f[2] - r[2]*f[1], r[2]*f[0] - r[0]*f[2], r[0]*f[1] - r[1]*f[0]];
  const fp = Math.min(w, h) * 1.45;
  return v => {
    const d = [v[0]-cam[0], v[1]-cam[1], v[2]-cam[2]];
    const zc = d[0]*f[0] + d[1]*f[1] + d[2]*f[2];
    const depth = CAM.orthographic ? CAM.dist : zc;
    return [w/2 + fp * (d[0]*r[0] + d[1]*r[1]) / depth, h/2 - fp * (d[0]*u[0] + d[1]*u[1] + d[2]*u[2]) / depth, zc];
  };
}
function draw3d(fr, fs) {
  const cv = $("c3"), w = cv.clientWidth, h = cv.clientHeight; if (!w || !h) return;
  const dpr = window.devicePixelRatio || 1;
  if (cv.width !== Math.round(w*dpr) || cv.height !== Math.round(h*dpr)) { cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr); }
  const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
  const P = makeProj(w, h), K = {};
  ["box","bowl","hand","goal","arm","past","future","ink","line","muted","warn","bad","table","card"].forEach(n => K[n] = css(n));
  const line = (a, b, c, lw, al, dash) => { const p = P(a), q = P(b); g.globalAlpha = al ?? 1; g.strokeStyle = c; g.lineWidth = lw; g.setLineDash(dash || []); g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke(); g.globalAlpha = 1; g.setLineDash([]); };
  const poly = (pts, fill, stroke, lw, al, dash) => { g.beginPath(); pts.forEach((v, k) => { const p = P(v); k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); }); g.closePath(); g.globalAlpha = al ?? 1; if (fill) { g.fillStyle = fill; g.fill(); } if (stroke) { g.strokeStyle = stroke; g.lineWidth = lw || 1; g.setLineDash(dash || []); g.stroke(); } g.globalAlpha = 1; g.setLineDash([]); };
  const txt = (v, s, c, size, anchor) => { const p = P(v); g.fillStyle = c; g.font = `${size || 11}px system-ui,sans-serif`; g.textAlign = anchor || "left"; g.fillText(s, p[0] + 6, p[1] - 6); };
  const ring = (c, rad, z, n = 32) => Array.from({ length: n }, (_, k) => [c[0] + rad*Math.cos(2*Math.PI*k/n), c[1] + rad*Math.sin(2*Math.PI*k/n), z]);
  const TZ = 0.42;

  // bàn + lưới
  const [TX, TY] = S.table;
  poly([[-TX,-TY,TZ],[TX,-TY,TZ],[TX,TY,TZ],[-TX,TY,TZ]], K.table, K.muted, 1, .55);
  for (let v = -0.4; v <= 0.401; v += 0.1) { line([v,-TY,TZ],[v,TY,TZ], K.muted, .5, .3); }
  for (let v = -0.4; v <= 0.401; v += 0.1) { line([-TX,v,TZ],[TX,v,TZ], K.muted, .5, .3); }
  // bát: đáy, miệng, thành, vòng ngưỡng
  const b = S.bowl, bz1 = b[2] + S.bowl_height;
  S.bowl_geoms.forEach(g => {                           // đáy (đĩa) + các tấm tường hẹp của thành bát, vẽ khung dây theo đúng kích thước model
    if (g.type === "cylinder") { poly(ring(g.pos, g.size[0], g.pos[2] + g.size[1]), K.bowl, K.bowl, 1, .3); return; }
    const V = wallCorners3(g);
    poly([V[4], V[5], V[7], V[6]], K.bowl, null, 0, .12);                          // mặt trên của tấm tường
    EDGES3.forEach(([p, q]) => line(V[p], V[q], K.bowl, 1, .7));
  });
  poly(ring(b, S.bowl_radius, bz1), null, K.bowl, 1, .8, [4, 3]);
  if (reachOn()) {                                      // một vùng màu nhạt, cắt theo mặt bàn
    const R = S.reach, bc = [R.base[0], R.base[1], TZ + 0.001], hh = R.r_step / 2;
    const path = pts => pts.forEach((v, n) => { const q = P(v); n ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1]); });
    g.save();
    g.beginPath(); path([[-TX, -TY, TZ], [TX, -TY, TZ], [TX, TY, TZ], [-TX, TY, TZ]]); g.closePath(); g.clip();
    R.bands[0].free.forEach(([lo, hi]) => {
      g.beginPath(); path(ring(bc, hi + hh, TZ + 0.001, 72)); g.closePath();
      if (lo > 0.03) { path(ring(bc, lo - hh, TZ + 0.001, 72)); g.closePath(); }
      g.fillStyle = "#0e9aa7"; g.globalAlpha = .17; g.fill("evenodd"); g.globalAlpha = 1;
    });
    g.restore();
  }

  // hộp ban đầu (nét đứt) và mũi tên lệch
  const cube = (c, hs) => { const V = []; for (const sx of [-1,1]) for (const sy of [-1,1]) for (const sz of [-1,1]) V.push([c[0]+sx*hs, c[1]+sy*hs, c[2]+sz*hs]); return V; };
  const EDGES = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],[0,4],[1,5],[2,6],[3,7]];
  const b0 = fs[0].box;
  const V0 = boxV(b0, bq(fs[0]), S.box_half);
  EDGES.forEach(([p, q]) => line(V0[p], V0[q], K.box, 1, .5, [3, 3]));
  if (Math.hypot(fr.box[0]-b0[0], fr.box[1]-b0[1]) > 0.004) line(b0, fr.box, K.bad, 2, 1, [2, 2]);

  // cánh tay (xương) từ danh sách điểm: 6 khớp + điểm giữa 2 ngón
  // lưới tam giác THẬT của cánh tay (vẽ xa trước), shading theo hướng pháp tuyến
  const LIGHT = (() => { const l = [0.35, -0.45, 0.82], n = Math.hypot(...l); return l.map(v => v / n); })();
  const drawMesh = f => {
    const W3 = armWorld(f), tris = [];
    for (const nm of Object.keys(S.arm_mesh)) {
      const w = W3[nm], F = S.arm_mesh[nm].f, n = w.length / 3, sc = new Float64Array(n * 3);
      for (let k = 0; k < n; k++) { const q = P([w[3*k], w[3*k+1], w[3*k+2]]); sc[3*k] = q[0]; sc[3*k+1] = q[1]; sc[3*k+2] = q[2]; }
      const faceColor = meshColors(nm, f);
      for (let t = 0; t < F.length; t += 3) {
        const i0 = F[t], i1 = F[t+1], i2 = F[t+2];
        const ux = w[3*i1]-w[3*i0], uy = w[3*i1+1]-w[3*i0+1], uz = w[3*i1+2]-w[3*i0+2];
        const vx = w[3*i2]-w[3*i0], vy = w[3*i2+1]-w[3*i0+1], vz = w[3*i2+2]-w[3*i0+2];
        let nx = uy*vz - uz*vy, ny = uz*vx - ux*vz, nz = ux*vy - uy*vx; const nl = Math.hypot(nx, ny, nz) || 1;
        const lam = Math.abs((nx*LIGHT[0] + ny*LIGHT[1] + nz*LIGHT[2]) / nl), sh = 0.42 + 0.58 * lam;
        tris.push([(sc[3*i0+2] + sc[3*i1+2] + sc[3*i2+2]) / 3, i0, i1, i2, sc, faceColor(t / 3), sh, .95]);
      }
    }
    // hộp hiện tại cùng được sắp theo độ sâu với cánh tay, để tay che / bị hộp che đúng chỗ
    { const Vb = boxV(f.box, bq(f), S.box_half), sb = new Float64Array(24);
      Vb.forEach((v, k) => { const q = P(v); sb[3*k] = q[0]; sb[3*k+1] = q[1]; sb[3*k+2] = q[2]; });
      [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]].forEach(([a, b, c, d], k) => {
        const ctr = (sb[3*a+2] + sb[3*b+2] + sb[3*c+2] + sb[3*d+2]) / 4, sh = [0.55, 1, 0.75, 0.9, 0.65, 0.8][k];
        tris.push([ctr, a, b, c, sb, [217, 54, 62], sh, 0.95], [ctr, a, c, d, sb, [217, 54, 62], sh, 0.95]);
      }); }
    tris.sort((p, q) => q[0] - p[0]);
    g.lineJoin = "round"; g.lineWidth = .6;
    for (const [, i0, i1, i2, sc, base, sh, al] of tris) {
      g.globalAlpha = al;
      const col = `rgb(${(base[0]*sh)|0},${(base[1]*sh)|0},${(base[2]*sh)|0})`;
      g.fillStyle = col; g.strokeStyle = col;
      g.beginPath(); g.moveTo(sc[3*i0], sc[3*i0+1]); g.lineTo(sc[3*i1], sc[3*i1+1]); g.lineTo(sc[3*i2], sc[3*i2+1]); g.closePath(); g.fill(); g.stroke();
    }
    g.globalAlpha = 1;
  };
  // cánh tay (xương) từ danh sách điểm: 6 khớp + điểm giữa 2 ngón; cánh tay chính vẽ thêm lưới thật bên dưới
  const pts = f => [...f.arm, f.ee];
  const drawArm = (f, lw, al, col, main) => {
    const A = pts(f);
    if (main) drawMesh(f);
    for (let k = 0; k < A.length - 1; k++) {
      const nm = S.arm_bodies[k], hit = main && f.ct.includes(nm), tb = main && f.tb.includes(nm), bw = main && touchBowl(f, nm);
      g.save();
      g.lineCap = "round";
      const color = hit ? K.bad : tb ? K.warn : bw ? BOWLC : (col || K.arm);
      line(A[k], A[k+1], color, main ? 2.4 : lw, main ? .95 : al);
      g.restore();
      if (hit || tb || bw) { const m = [0,1,2].map(a => (A[k][a] + A[k+1][a]) / 2); txt(m, nm + (hit ? " CHẠM HỘP" : tb ? " chạm bàn" : " chạm BÁT"), hit ? K.bad : tb ? K.warn : BOWLC, 11, "left"); }
    }
    if (main) {
      for (let k = 1; k <= 5; k++) {
        const q = P(A[k]); g.fillStyle = K.card; g.strokeStyle = K.ink; g.lineWidth = 1.3; g.beginPath(); g.arc(q[0], q[1], 8, 0, 7); g.fill(); g.stroke();
        g.fillStyle = K.ink; g.font = "bold 10px system-ui,sans-serif"; g.textAlign = "center"; g.fillText(k, q[0], q[1] + 3.5);
        g.font = "10px system-ui,sans-serif"; g.textAlign = "left"; g.fillStyle = K.muted; g.fillText(JN[k-1], q[0] + 10, q[1] + 3.5);
      }
    }
  };

  // quá khứ / tương lai (cánh tay mờ + đường đi của tay), dùng chung tuỳ chọn với 2 hình 2D
  const n = fs.length - 1;
  if (OPT.past && i > 0) {
    const st = pastStart(), idx = ghostIdx(i, st);
        idx.slice().reverse().forEach(k => { const t = Math.abs(k - i) / Math.max(1, Math.abs(st - i)); drawArm(fs[k], 4, .7 - .35*t, K.past); });
  }
  if (OPT.future && i < n) {
    const e = futureEnd(), idx = ghostIdx(i, e);
    idx.slice().reverse().forEach(k => { const t = Math.abs(k - i) / Math.max(1, Math.abs(e - i)); drawArm(fs[k], 4, .75 - .35*t, K.future); });
  }

  // cánh tay hiện tại (trên cùng), tay, hình chiếu xuống bàn và đích
  drawArm(fr, 7, .95, K.arm, true);
  // đường đi của tay vẽ SAU cánh tay hiện tại để không bị mesh che
  if (OPT.path && OPT.past && i > 0) { g.strokeStyle = K.past; g.lineWidth = 3; g.globalAlpha = .95; g.beginPath(); fs.slice(pastStart(), i + 1).forEach((f, k) => { const p = P(f.ee); k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); }); g.stroke(); g.globalAlpha = 1; }
  if (OPT.path && OPT.future && i < n) { g.strokeStyle = K.future; g.lineWidth = 3.2; g.setLineDash([6, 3]); g.beginPath(); fs.slice(i, futureEnd() + 1).forEach((f, k) => { const p = P(f.ee); k ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); }); g.stroke(); g.setLineDash([]); }
  const hz = [fr.ee[0], fr.ee[1], TZ];
  line(fr.ee, hz, K.hand, 1, .8, [2, 3]);
  poly(ring(hz, 0.008, TZ, 12), K.hand, null, 0, .5);
  { const q = P(fr.ee); g.strokeStyle = K.hand; g.lineWidth = 2.5; g.beginPath(); g.moveTo(q[0]-8, q[1]); g.lineTo(q[0]+8, q[1]); g.moveTo(q[0], q[1]-8); g.lineTo(q[0], q[1]+8); g.stroke(); }
  if (fr.tg) {
    line(fr.ee, fr.tg, K.warn, 2, 1);
    const q = P(fr.tg); g.strokeStyle = K.goal; g.lineWidth = 2.5; g.beginPath(); g.arc(q[0], q[1], 8, 0, 7); g.stroke();
    line(fr.tg, [fr.tg[0], fr.tg[1], TZ], K.goal, 1, .6, [2, 3]);
  }

  // la bàn trục x (đỏ), y (xanh lá), z (xanh dương) ở góc dưới trái; xoay theo camera
  {
    const cp = Math.cos(CAM.pitch), sp = Math.sin(CAM.pitch), cy = Math.cos(CAM.yaw), sy = Math.sin(CAM.yaw);
    const f = [-cp*cy, -cp*sy, -sp]; let r = [f[1], -f[0], 0]; const rl = Math.hypot(r[0], r[1]) || 1; r = [r[0]/rl, r[1]/rl, 0];
    const u = [r[1]*f[2] - r[2]*f[1], r[2]*f[0] - r[0]*f[2], r[0]*f[1] - r[1]*f[0]];
    const ox = 40, oy = h - 36, L = 30;
    g.fillStyle = K.card; g.globalAlpha = .65; g.beginPath(); g.arc(ox, oy, L + 12, 0, 7); g.fill(); g.globalAlpha = 1;
    [[[1,0,0],"#d33","x"],[[0,1,0],"#2a2","y"],[[0,0,1],"#38e","z"]].forEach(([d, c, n]) => {
      const sx = d[0]*r[0] + d[1]*r[1], sy2 = -(d[0]*u[0] + d[1]*u[1] + d[2]*u[2]);
      g.strokeStyle = c; g.fillStyle = c; g.lineWidth = 2.4; g.beginPath(); g.moveTo(ox, oy); g.lineTo(ox + L*sx, oy + L*sy2); g.stroke();
      g.font = "bold 12px system-ui,sans-serif"; g.textAlign = "center"; g.fillText(n, ox + (L + 8)*sx, oy + (L + 8)*sy2 + 4);
    });
  }
}

/* ---------- camera trong model: dựng lại góc nhìn của camera đầu kẹp (arm_cam) bằng cùng bộ vẽ mesh ---------- */
function camWorld(cam, fr) {                             // tư thế world của camera: cột của R là trục x (phải), y (lên), z (sau lưng)
  if (!cam.body) return { R: cam.R, p: cam.pos };
  const T = fkPose(fr)[cam.body];
  return { R: mmul(T.R, cam.R), p: mvec(T.R, cam.pos).map((v, j) => v + T.p[j]) };
}
function renderCam(cv, name, fr) {
  const cam = (S.cams || {})[name]; if (!cam) return;
  const g = cv.getContext("2d"), w = cv.width, h = cv.height, f = (h / 2) / Math.tan(cam.fovy * Math.PI / 360);
  const { R, p: c } = camWorld(cam, fr);
  const toCam = v => { const d = [v[0]-c[0], v[1]-c[1], v[2]-c[2]];
    return [R[0][0]*d[0] + R[1][0]*d[1] + R[2][0]*d[2], R[0][1]*d[0] + R[1][1]*d[1] + R[2][1]*d[2], R[0][2]*d[0] + R[1][2]*d[1] + R[2][2]*d[2]]; };
  const scr = q => [w/2 + f * q[0] / -q[2], h/2 - f * q[1] / -q[2]];
  g.fillStyle = "#cfd8e3"; g.fillRect(0, 0, w, h);
  g.fillStyle = "#b9bec6"; g.fillRect(0, h * 0.0, w, 0);                       // nền (sàn vẽ bằng mặt bàn bên dưới)
  const items = [];                                                                // [độ sâu, các đỉnh cam, màu, sáng]
  const poly = (verts, col, sh, depthBias) => { const q = verts.map(toCam); if (q.some(v => v[2] > -0.02)) return;
    items.push([-(q.reduce((s, v) => s + v[2], 0) / q.length) + (depthBias || 0), q, col, sh]); };
  const quad = (a, b2, c2, d2, col, sh, bias) => poly([a, b2, c2, d2], col, sh, bias);
  // mặt bàn: chia lưới để không bị cắt sai khi camera ở gần
  const [TX, TY] = S.table, TZ = 0.42, N = 6;
  for (let i2 = 0; i2 < N; i2++) for (let j = 0; j < N; j++) {
    const x0 = -TX + 2*TX*i2/N, x1 = -TX + 2*TX*(i2+1)/N, y0 = -TY + 2*TY*j/N, y1 = -TY + 2*TY*(j+1)/N;
    quad([x0,y0,TZ],[x1,y0,TZ],[x1,y1,TZ],[x0,y1,TZ], [196,140,86], 1 - 0.06 * ((i2 + j) % 2), 100);
  }
  const FACES = [[0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5]], SH = [0.55, 1, 0.75, 0.9, 0.65, 0.8];
  S.bowl_geoms.forEach(gm => { if (gm.type === "cylinder") { const ring = Array.from({length: 18}, (_, q) => { const t = q/18*2*Math.PI; return [gm.pos[0] + gm.size[0]*Math.cos(t), gm.pos[1] + gm.size[0]*Math.sin(t), gm.pos[2] + gm.size[1]]; }); poly(ring, [60,110,210], .9, 50); return; }
    const V = wallCorners3(gm); FACES.forEach((F2, k2) => poly(F2.map(q => V[q]), [60,110,210], SH[k2])); });
  const Vb = boxV(fr.box, bq(fr), S.box_half); FACES.forEach((F2, k2) => poly(F2.map(q => Vb[q]), [217,54,62], SH[k2]));
  const W3 = armWorld(fr), L = [0.35, -0.45, 0.82], ln = Math.hypot(...L);
  for (const nm of Object.keys(S.arm_mesh)) { const wv = W3[nm], Fi = S.arm_mesh[nm].f, n = wv.length / 3, cc = new Array(n);
    for (let k2 = 0; k2 < n; k2++) cc[k2] = toCam([wv[3*k2], wv[3*k2+1], wv[3*k2+2]]);
    const faceColor = meshColors(nm, fr);
    for (let t = 0; t < Fi.length; t += 3) { const a2 = cc[Fi[t]], b2 = cc[Fi[t+1]], c2 = cc[Fi[t+2]];
      if (a2[2] > -0.02 || b2[2] > -0.02 || c2[2] > -0.02) continue;
      const ux = b2[0]-a2[0], uy = b2[1]-a2[1], uz = b2[2]-a2[2], vx = c2[0]-a2[0], vy = c2[1]-a2[1], vz = c2[2]-a2[2];
      const nx = uy*vz - uz*vy, ny = uz*vx - ux*vz, nz = ux*vy - uy*vx, nl = Math.hypot(nx, ny, nz) || 1;
      items.push([-(a2[2] + b2[2] + c2[2]) / 3, [a2, b2, c2], faceColor(t / 3), 0.45 + 0.55 * Math.abs(nz / nl)]); } }
  items.sort((p1, p2) => p2[0] - p1[0]);
  g.lineJoin = "round"; g.lineWidth = .6;
  for (const [, q, col, sh] of items) {
    const cs = `rgb(${(col[0]*sh)|0},${(col[1]*sh)|0},${(col[2]*sh)|0})`; g.fillStyle = cs; g.strokeStyle = cs; g.beginPath();
    q.forEach((v, k2) => { const s2 = scr(v); k2 ? g.lineTo(s2[0], s2[1]) : g.moveTo(s2[0], s2[1]); }); g.closePath(); g.fill(); g.stroke();
  }
  g.fillStyle = "#000a"; g.fillRect(0, 0, name.length * 6.5 + 10, 16); g.fillStyle = "#fff"; g.font = "11px system-ui"; g.fillText(name, 4, 12);
}
const hexRgbCam = h => [1, 3, 5].map(k => parseInt(h.slice(k, k + 2), 16));
let SHOW_CAM = true;
try { SHOW_CAM = localStorage.getItem("showArmCam") !== "0"; } catch (e) {}
function applyCamToggle() { $("cameraFeeds").hidden = !SHOW_CAM; $("bCam").setAttribute("aria-pressed", String(SHOW_CAM));  }
function drawCams(fr) {
  if (!SHOW_CAM) return;
  renderCam($("camA"), "arm_cam", fr);
}

/* ---------- điều khiển ---------- */
function render() {
  const fs = F(), fr = fs[i];
  drawTop(fr, fs); drawSide(fr, fs); draw3d(fr, fs); drawCams(fr); drawInfo(fr, fs); drawChk(fr);
  $("rng").value = i;
  $("stepCount").value = `${i} / ${fs.length - 1} · ${(i * STEP_DT).toFixed(2)} / ${((fs.length - 1) * STEP_DT).toFixed(2)} s thực tế`;
  $("tlh").style.left = tlPos(i) + "%"; $("tlh").setAttribute("aria-valuenow", i); $("tlh").setAttribute("aria-valuemax", fs.length - 1);
  $("tlh").setAttribute("aria-valuetext", "bước " + i + ", phase " + fr.ph);
  const cur = $("logw").querySelector("tr.cur"); if (cur) cur.classList.remove("cur");
  const row = $("logw").querySelector(`tr[data-k="${i}"]`);
  if (row) { row.classList.add("cur"); const w = $("logw"), t = row.offsetTop; if (t < w.scrollTop + 30 || t > w.scrollTop + w.clientHeight - 30) w.scrollTop = t - w.clientHeight / 2; }
}
function go(k) { i = Math.max(0, Math.min(F().length - 1, k)); render(); }
function nextPhase() {
  const fs = F(); let k = i + 1;
  while (k < fs.length - 1 && fs[k].ph === fs[k+1].ph) k++;
  go(Math.min(k, fs.length - 1));
}
function prevPhase() {
  const fs = F(); let k = i - 1;
  while (k > 0 && fs[k-1].ph === fs[k].ph) k--;
  go(Math.max(0, k - 1));
}
const PLAY_ICON = '<svg class="h-[14px] w-[14px]" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="6 3 20 12 6 21 6 3"/></svg>', PAUSE_ICON = '<svg class="h-[14px] w-[14px]" viewBox="0 0 24 24" fill="currentColor" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="14" y="4" width="4" height="16" rx="1"/><rect x="6" y="4" width="4" height="16" rx="1"/></svg>';
function stop() { if (timer) { clearInterval(timer); timer = null; } $("bPlay").innerHTML = PLAY_ICON; $("bPlay").setAttribute("aria-label", "Phát"); }
function play() {
  if (timer) return stop();
  if (i >= F().length - 1) i = 0;
  $("bPlay").innerHTML = PAUSE_ICON; $("bPlay").setAttribute("aria-label", "Dừng");
  // Phát theo ĐỒNG HỒ thật: khung hình = thời gian đã trôi × tốc độ / STEP_DT, nên không bị chậm đi khi máy vẽ không kịp
  let t0 = performance.now(), i0 = i;
  timer = setInterval(() => {
    const k = Math.min(F().length - 1, i0 + Math.floor((performance.now() - t0) / 1000 * speed / STEP_DT));
    if (k !== i) go(k);
    if (i >= F().length - 1) stop();
  }, 16);
}
function loadEp(k) {
  stop(); ep = k; i = 0;
  if (EPS[ep].bowl) { S.bowl = EPS[ep].bowl; S.bowl_geoms = EPS[ep].bowl_geoms; }   // mỗi ván có thể có bát ở chỗ khác
  const e = EPS[k];
  $("rng").max = e.frames.length - 1;
  $("res").innerHTML = `<span class="badge ${e.success ? "ok" : "bad"}">${e.success ? "THÀNH CÔNG" : "THẤT BẠI"}</span> <strong>${e.steps} bước tổng</strong>`;
  $("sel").value = k; $("epPos").textContent = `${k + 1}/${EPS.length}`;
  $("tags").innerHTML = e.tags ? e.tags.filter(x => x !== "THÀNH CÔNG").map(x => `<span class="badge ${/HỎNG|timeout/.test(x) ? "bad" : "warn"}" style="margin-right:3px">${x}</span>`).join("") : "";
  buildLog(e.frames); buildTimeline(e.frames); render();
}
$("showReach").onchange = () => render();
$("sel").onchange = () => loadEp(+$("sel").value);
$("sPrev").onclick = () => loadEp((ep - 1 + EPS.length) % EPS.length);
$("sNext").onclick = () => loadEp((ep + 1) % EPS.length);
$("spd").onchange = () => { speed = +$("spd").value; if (timer) { stop(); play(); } };   // play() lấy lại mốc thời gian từ khung đang xem
$("rng").oninput = () => go(+$("rng").value);
$("bFirst").onclick = () => go(0);
$("bPrev").onclick = () => go(i - 1);
$("bNext").onclick = () => go(i + 1);
$("bPhase").onclick = nextPhase;
$("bPP").onclick = prevPhase;
$("bPlay").onclick = play;
$("oPast").onchange = e => { OPT.past = e.target.checked; render(); };
$("oFut").onchange = e => { OPT.future = e.target.checked; render(); };
$("oPath").onchange = e => { OPT.path = e.target.checked; render(); };
$("oHz").onchange = e => { OPT.hz = e.target.value; $("oEv").disabled = OPT.hz === "1"; render(); };   // phạm vi 1 bước thì chỉ có 1 cánh tay mờ, khoảng cách không còn ý nghĩa
$("oEv").disabled = true;
$("oEv").onchange = e => { OPT.every = +e.target.value; render(); };
// điều khiển camera 3D
const redraw3 = () => draw3d(F()[i], F());
const setCam = (yaw, pitch, dist = CAM0.dist, orthographic = false) => {
  Object.assign(CAM, { yaw, pitch, dist, orthographic });
  redraw3();
};
$("v1").onclick = () => setCam(CAM0.yaw, CAM0.pitch, CAM0.dist);
$("v2").onclick = () => setCam(Math.PI / 2, 0, CAM0.dist, true);
$("vSide").onclick = () => setCam(0, 0, CAM0.dist, true);
$("vCams").onclick = () => setCam(CAM0.yaw, CAM0.pitch, CAM0.dist, true);
$("v3").onclick = () => setCam(-Math.PI / 2, Math.PI / 2, CAM0.dist, true);
{
  const base = S.cam_tilt_deg || 0, sl = $("camTilt");
  const apply = v => { CAM_TILT_EXTRA = (v - base) * Math.PI / 180; $("camTiltV").textContent = v + "°";
    EPS.forEach(e => e.frames.forEach(f => { delete f._pose; delete f._w; })); render(); };
  let saved = null; try { saved = localStorage.getItem("camTilt"); } catch (e) {}
  sl.value = saved !== null ? saved : base;
  sl.oninput = () => { apply(+sl.value); try { localStorage.setItem("camTilt", sl.value); } catch (e) {} };
  sl.ondblclick = () => { sl.value = base; sl.oninput(); };
  if (+sl.value !== base) apply(+sl.value); else $("camTiltV").textContent = base + "°";
}
$("bCam").onclick = () => { SHOW_CAM = !SHOW_CAM; try { localStorage.setItem("showArmCam", SHOW_CAM ? "1" : "0"); } catch (e) {} applyCamToggle(); render(); };
applyCamToggle();
{
  const cv = $("c3"); let drag = null;
  cv.onpointerdown = e => { drag = { x: e.clientX, y: e.clientY }; cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing"; };
  cv.onpointerup = () => { drag = null; cv.style.cursor = "grab"; };
  cv.onpointermove = e => {
    if (!drag) return;
    CAM.yaw -= (e.clientX - drag.x) * 0.008;
    CAM.pitch = Math.max(-0.2, Math.min(1.5, CAM.pitch + (e.clientY - drag.y) * 0.008));
    drag = { x: e.clientX, y: e.clientY }; redraw3();
  };
  cv.onwheel = e => { e.preventDefault(); CAM.dist = Math.max(0.3, Math.min(2.5, CAM.dist * Math.exp(e.deltaY * 0.001))); redraw3(); };
  cv.ondblclick = () => setCam(CAM0.yaw, CAM0.pitch, CAM0.dist);
  new ResizeObserver(redraw3).observe(cv);
}
document.addEventListener("keydown", e => {
  if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey ||
      e.target.closest?.("input, select, textarea, button, summary, a, [contenteditable]")) return;
  if (e.key === "[") $("sPrev").click();
  else if (e.key === "]") $("sNext").click();
  else if (e.key === "ArrowRight") go(i + 1);
  else if (e.key === "ArrowLeft") go(i - 1);
  else if (e.key === "ArrowDown") nextPhase();
  else if (e.key === "ArrowUp") prevPhase();
  else if (e.key === " ") { e.preventDefault(); play(); }
  else if (e.key === "Home") go(0);
  else if (e.key === "End") go(F().length - 1);
  else return;
  e.preventDefault();
});
// Nhớ ván đang xem và khung hình trong localStorage để F5 không mất. Khoá theo TÊN ván (CASE 1... hoặc seed N),
// không theo số thứ tự, vì danh sách đổi sau mỗi lần chạy select_seeds.
const epKey = e => String(e.name || "seed " + e.seed);
const store = {
  get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} },
};
const saveView = () => { store.set("replayEp", epKey(EPS[ep])); store.set("replayFrame", String(i)); };
{
  const saved = store.get("replayEp");
  let def = saved ? EPS.findIndex(e => epKey(e) === saved) : -1;
  const restored = def >= 0;
  if (def < 0) def = EPS.findIndex(e => /^CASE 1/.test(e.name || ""));   // mặc định: hộp XA nhất
  loadEp(def >= 0 ? def : 0);
  if (restored) go(Math.min(+store.get("replayFrame") || 0, F().length - 1));
}
// lưu mỗi khi đổi ván hoặc khung hình
{ const r = render; render = function () { r.apply(this, arguments); saveView(); }; }
