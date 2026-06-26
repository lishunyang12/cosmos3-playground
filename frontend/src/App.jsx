import { useEffect, useMemo, useRef, useState } from "react";
import { getConfig, generate, getJob, jobContentUrl, requestPreview, exampleAction, validate, rolloutStart, rolloutStatus, rolloutContentUrl } from "./api.js";
import {
  IconAtom, IconSparkles, IconEye, IconUpload, IconAlert,
  IconChevron, IconReset, IconDownload,
} from "./icons.jsx";

// Simple view = the forgiving, high-quality tasks that "just work"; everything else
// (action/policy/transfer/grounding…) is expert/experimental, shown only in Advanced.
const SIMPLE_TASKS = new Set(["t2i", "t2v", "i2v", "caption", "vqa"]);

function fmt(sec) {
  if (sec == null || isNaN(sec)) return "0s";
  sec = Math.max(0, Math.round(sec));
  return sec < 60 ? `${sec}s` : `${Math.floor(sec / 60)}m${String(sec % 60).padStart(2, "0")}s`;
}

// Inverse-dynamics (and any action-output mode) returns a [T, D] trajectory, not a clip.
// Surface it as a motion-profile plot + a numeric table + a full-data download.
function ActionTrajectory({ action, title }) {
  const shape = action.shape || [];
  const d = action.data;
  const cols = shape[1] || (Array.isArray(d?.[0]) ? d[0].length : 1);
  let rows = [];
  if (Array.isArray(d) && Array.isArray(d[0])) rows = d;
  else if (Array.isArray(d)) for (let i = 0; i < d.length; i += cols) rows.push(d.slice(i, i + cols));
  const n = rows.length;

  // per-step L2 magnitude — the overall motion profile over time
  const mags = rows.map((r) => Math.sqrt(r.reduce((s, x) => s + Number(x) ** 2, 0)));
  const W = 520, H = 60, pad = 4;
  const hi = Math.max(...mags, 1e-6), lo = Math.min(...mags, 0);
  const pts = mags.map((m, i) => {
    const x = pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
    const y = H - pad - ((m - lo) / Math.max(1e-6, hi - lo)) * (H - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(" ");
  const preview = rows.slice(0, 14);

  function download() {
    const payload = { shape, dtype: action.dtype, action_mode: action.action_mode, domain_id: action.domain_id, data: rows };
    const url = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url; a.download = "action_trajectory.json"; a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="trajectory">
      <div className="traj-head">
        <span className="traj-title">{title || (action.action_mode === "inverse_dynamics" ? "Recovered action trajectory (model output)" : "Action plan (input)")}</span>
        <span className="traj-meta">{shape.join(" × ")} · domain {action.domain_id}</span>
      </div>
      <svg className="sparkline" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" aria-hidden="true">
        <polyline points={pts} fill="none" />
      </svg>
      <div className="traj-cap">curve = per-step motion magnitude ‖aₜ‖ over {n} steps · table = raw action value per dimension</div>
      <div className="traj-table-wrap">
        <table className="traj-table">
          <thead><tr><th>t</th>{Array.from({ length: cols }, (_, i) => <th key={i}>d{i}</th>)}</tr></thead>
          <tbody>
            {preview.map((r, i) => (
              <tr key={i}><td className="t">{i}</td>{r.map((x, j) => <td key={j}>{Number(x).toFixed(3)}</td>)}</tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="traj-foot">
        showing {preview.length} of {n} steps · <a onClick={download}><IconDownload /> full trajectory (JSON)</a>
      </div>
    </div>
  );
}

// Round-trip validation for forward dynamics: run inverse dynamics on the generated video
// (same domain) and score the recovered action against the original plan. A high consistency
// means the world model actually obeyed the action it was given.
function ValidationCard({ state, onRun }) {
  const s = state?.data?.score;
  const pct = s?.consistency_pct;
  const tone = pct == null ? "" : pct >= 90 ? " good" : pct >= 70 ? " ok" : " poor";
  const perCh = s?.per_channel_mae || [];
  const hi = Math.max(...perCh, 1e-6);
  return (
    <div className={"validation" + tone}>
      <div className="val-head">
        <span className="val-title">Round-trip validation</span>
        <span className="val-sub">forward → inverse · does the world obey the action?</span>
        <button type="button" className="val-run" disabled={state?.status === "running"} onClick={onRun}>
          {state?.status === "running" ? "Checking…" : s ? "Re-check" : "Validate output"}
        </button>
      </div>
      {state?.error && <div className="val-error"><IconAlert /> {state.error}</div>}
      {state?.status === "running" && <div className="val-note">running inverse dynamics on the generated video in the {state?.domain || "same"} domain…</div>}
      {s && (
        <>
          <div className="val-scores">
            <div className="val-metric big"><span className="val-num">{pct}%</span><span className="val-lbl">consistency</span></div>
            <div className="val-metric"><span className="val-num">{s.cosine}</span><span className="val-lbl">cosine</span></div>
            <div className="val-metric"><span className="val-num">{s.mae}</span><span className="val-lbl">mean abs err</span></div>
            <div className="val-metric"><span className="val-num">{(s.shape || []).join("×")}</span><span className="val-lbl">{state.data.domain}</span></div>
          </div>
          <div className="val-channels" aria-label="per-channel error">
            {perCh.map((v, i) => (
              <span key={i} className="val-ch" title={`d${i}: ${v}`}>
                <span className="val-ch-bar" style={{ height: `${Math.max(2, (v / hi) * 100)}%` }} />
              </span>
            ))}
          </div>
          <div className="val-cap">per-channel recovery error (taller = the model drifted from your plan on that dimension)</div>
        </>
      )}
    </div>
  );
}

// How the request executes on the server: the deployment's parallel layout plus the
// computation DAG (inputs → encoders → denoiser → decoders → outputs), each compute
// node tagged with the parallel dimensions acting on it.
const DIM_LABEL = { cfg: "CFG", ulysses: "USP", ring: "Ring", tp: "TP", pp: "PP", dp: "DP", vae: "VAE" };

// TensorBoard-style layered DAG: longest-path depth = column, op-typed nodes, namescope
// clusters drawn behind, cubic-bezier edges labelled with the tensor shape they carry.
function ComputationGraph({ graph, dims }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [fs, setFs] = useState(false);
  const vpRef = useRef(null);
  const panRef = useRef(null);
  const dimsRef = useRef({ W: 1, H: 1 });
  const zoomRef = useRef(zoom);
  const panValRef = useRef(pan);
  zoomRef.current = zoom;
  panValRef.current = pan;
  const clampZ = (z) => Math.min(3, Math.max(0.4, z));

  // Auto-fit: scale the graph to fill the frame (centered) on load / when it changes / on
  // entering fullscreen — so it never sits tiny in an empty box.
  useEffect(() => {
    const vp = vpRef.current;
    if (!vp) return;
    const { W, H } = dimsRef.current;
    const z = Math.min(3, Math.max(0.4, Math.min((vp.clientWidth - 24) / W, (vp.clientHeight - 24) / H)));
    setZoom(z);
    setPan({ x: Math.max(8, (vp.clientWidth - W * z) / 2), y: Math.max(8, (vp.clientHeight - H * z) / 2) });
  }, [graph, fs]);

  useEffect(() => {
    if (!fs) return;
    const onKey = (e) => { if (e.key === "Escape") setFs(false); };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [fs]);

  // Native, non-passive wheel listener so preventDefault actually stops the browser's
  // ctrl/⌘+wheel page zoom — the graph zooms on its own, the page does not.
  useEffect(() => {
    const vp = vpRef.current;
    if (!vp) return;
    const onWheelNative = (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      const rect = vp.getBoundingClientRect();
      const vx = e.clientX - rect.left, vy = e.clientY - rect.top;  // cursor in viewport coords
      const z = zoomRef.current, p = panValRef.current;
      const nz = Math.min(3, Math.max(0.4, z * (e.deltaY < 0 ? 1.1 : 0.9)));
      if (nz === z) return;
      // keep the point under the cursor fixed while zooming
      setZoom(nz);
      setPan({ x: vx - (vx - p.x) * (nz / z), y: vy - (vy - p.y) * (nz / z) });
    };
    vp.addEventListener("wheel", onWheelNative, { passive: false });
    return () => vp.removeEventListener("wheel", onWheelNative);
  }, [fs]);

  const nodes = graph?.nodes || [];
  const edges = graph?.edges || [];
  if (!nodes.length) return null;

  const preds = {};
  nodes.forEach((n) => { preds[n.id] = []; });
  edges.forEach((e) => { if (preds[e.to]) preds[e.to].push(e.from); });
  const depth = {};
  const dep = (id) => {
    if (depth[id] != null) return depth[id];
    depth[id] = 0; // guard cycles
    const ps = preds[id] || [];
    depth[id] = ps.length ? Math.max(...ps.map(dep)) + 1 : 0;
    return depth[id];
  };
  nodes.forEach((n) => dep(n.id));

  const layers = {};
  nodes.forEach((n) => { (layers[depth[n.id]] ||= []).push(n); });
  const maxLayer = Math.max(...Object.keys(layers).map(Number));
  const rowsMax = Math.max(...Object.values(layers).map((c) => c.length));

  const NW = 150, NH = 60, COL = 212, ROW = 98, PADX = 16, PADTOP = 30, PADBOT = 16;
  const pos = {};
  for (let l = 0; l <= maxLayer; l++) {
    const col = layers[l] || [];
    const offY = (rowsMax * ROW - col.length * ROW) / 2;
    col.forEach((n, i) => { pos[n.id] = { x: PADX + l * COL, y: PADTOP + offY + i * ROW }; });
  }
  const W = PADX * 2 + maxLayer * COL + NW;
  const H = PADTOP + PADBOT + rowsMax * ROW;
  dimsRef.current = { W, H };  // expose to the auto-fit effect

  const clusters = (graph.scopes || []).map((s) => {
    const ps = nodes.filter((n) => n.scope === s.id).map((n) => pos[n.id]).filter(Boolean);
    if (!ps.length) return null;
    const CP = 12, LBL = 14;
    const minX = Math.min(...ps.map((p) => p.x)), minY = Math.min(...ps.map((p) => p.y));
    const maxX = Math.max(...ps.map((p) => p.x)) + NW, maxY = Math.max(...ps.map((p) => p.y)) + NH;
    return { id: s.id, label: s.label, x: minX - CP, y: minY - CP - LBL, w: maxX - minX + CP * 2, h: maxY - minY + CP * 2 + LBL };
  }).filter(Boolean);

  const edgePath = (a, b) => {
    const s = pos[a], t = pos[b];
    const x1 = s.x + NW, y1 = s.y + NH / 2, x2 = t.x, y2 = t.y + NH / 2, mx = (x1 + x2) / 2;
    return `M${x1},${y1} C${mx},${y1} ${mx},${y2} ${x2},${y2}`;
  };

  const fit = () => { const vp = vpRef.current; if (vp) { setZoom(clampZ((vp.clientWidth - 28) / W)); setPan({ x: 0, y: 0 }); } };
  const reset = () => { setZoom(1); setPan({ x: 0, y: 0 }); };
  const onDown = (e) => {
    if (e.button !== 0) return;
    panRef.current = { x: e.clientX, y: e.clientY, px: pan.x, py: pan.y };
    vpRef.current && vpRef.current.classList.add("grabbing");
  };
  const onMove = (e) => {
    const p = panRef.current; if (!p) return;
    setPan({ x: p.px + (e.clientX - p.x), y: p.py + (e.clientY - p.y) });
  };
  const endPan = () => { panRef.current = null; vpRef.current && vpRef.current.classList.remove("grabbing"); };

  const graphBody = (
    <>
      <div className="cgraph-toolbar">
        <button type="button" className="cgraph-fsbtn" onClick={() => setFs((f) => !f)}>{fs ? "Exit ⤡" : "Fullscreen ⤢"}</button>
      </div>
      <div className="cgraph-vp" ref={vpRef}
        onMouseDown={onDown} onMouseMove={onMove} onMouseUp={endPan} onMouseLeave={endPan}>
        <div className="cgraph-sizer">
          <div className="cgraph" style={{ width: W, height: H, transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`, transformOrigin: "top left" }}>
            {clusters.map((c) => (
              <div key={c.id} className="cscope" style={{ left: c.x, top: c.y, width: c.w, height: c.h }}>
                <span className="cscope-label">{c.label}</span>
              </div>
            ))}
            <svg className="cgraph-edges" width={W} height={H}>
              <defs>
                <marker id="cg-arrow" markerWidth="8" markerHeight="8" refX="6.5" refY="3" orient="auto">
                  <path d="M0,0 L6,3 L0,6 Z" />
                </marker>
              </defs>
              {edges.map((e, i) => (pos[e.from] && pos[e.to] ? (
                <path key={i} d={edgePath(e.from, e.to)} markerEnd="url(#cg-arrow)" />
              ) : null))}
            </svg>
            {edges.map((e, i) => {
              if (!pos[e.from] || !pos[e.to] || !e.shape) return null;
              const s = pos[e.from], t = pos[e.to];
              const x = (s.x + NW + t.x) / 2, y = (s.y + t.y) / 2 + NH / 2;
              return <span key={i} className="cedge-label" style={{ left: x, top: y }}>{e.shape}</span>;
            })}
            {nodes.map((n) => {
              const tags = (n.dims || []).filter((d) => (dims[d] || 1) > 1);
              const name = n.kind === "compute" ? n.op : n.label;
              return (
                <div key={n.id} className={"cnode " + n.kind}
                  style={{ left: pos[n.id].x, top: pos[n.id].y, width: NW, height: NH }}>
                  <span className="cnode-label">{name}</span>
                  {n.shape && <span className="cnode-shape">{n.shape}</span>}
                  {tags.length > 0 && (
                    <span className="cnode-tags">
                      {tags.map((d) => <span key={d} className="cnode-tag">{DIM_LABEL[d]}×{dims[d]}</span>)}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );

  return fs
    ? <div className="cgraph-fs">{graphBody}</div>
    : <div className="cgraph-host">{graphBody}</div>;
}

function ExecutionTopology({ topology, graph }) {
  const dims = topology?.dims || {};
  return (
    <div className="topology">
      <ComputationGraph graph={graph} dims={dims} />
    </div>
  );
}

// ---- lightweight markdown (the VLM answers in markdown) ----
function mdInline(text) {
  const out = [];
  let rest = String(text), key = 0;
  const re = /(\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`|\[([^\]]+)\]\(([^)]+)\))/;
  while (rest) {
    const m = re.exec(rest);
    if (!m) { out.push(rest); break; }
    if (m.index > 0) out.push(rest.slice(0, m.index));
    if (m[2] != null) out.push(<strong key={key++}>{m[2]}</strong>);
    else if (m[3] != null) out.push(<em key={key++}>{m[3]}</em>);
    else if (m[4] != null) out.push(<code key={key++}>{m[4]}</code>);
    else if (m[5] != null) out.push(<a key={key++} href={m[6]} target="_blank" rel="noreferrer">{m[5]}</a>);
    rest = rest.slice(m.index + m[0].length);
  }
  return out;
}

function Markdown({ text }) {
  const lines = String(text || "").replace(/\r\n/g, "\n").split("\n");
  const blocks = [];
  let i = 0, key = 0;
  const isBreak = (l) => /^(#{1,4}\s|```|\s*[-*]\s|\s*\d+\.\s|\s*>\s?)/.test(l);
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const buf = []; i++;
      while (i < lines.length && !/^```/.test(lines[i])) { buf.push(lines[i]); i++; }
      i++; blocks.push(<pre key={key++} className="md-code"><code>{buf.join("\n")}</code></pre>); continue;
    }
    const h = /^(#{1,4})\s+(.*)$/.exec(line);
    if (h) { const Tag = `h${Math.min(6, h[1].length + 2)}`; blocks.push(<Tag key={key++} className="md-h">{mdInline(h[2])}</Tag>); i++; continue; }
    if (/^\s*[-*]\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*[-*]\s+/, "")); i++; }
      blocks.push(<ul key={key++} className="md-ul">{items.map((t, j) => <li key={j}>{mdInline(t)}</li>)}</ul>); continue;
    }
    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) { items.push(lines[i].replace(/^\s*\d+\.\s+/, "")); i++; }
      blocks.push(<ol key={key++} className="md-ol">{items.map((t, j) => <li key={j}>{mdInline(t)}</li>)}</ol>); continue;
    }
    if (line.trim() === "") { i++; continue; }
    const buf = [line]; i++;
    while (i < lines.length && lines[i].trim() !== "" && !isBreak(lines[i])) { buf.push(lines[i]); i++; }
    blocks.push(<p key={key++} className="md-p">{mdInline(buf.join(" "))}</p>);
  }
  return <div className="md">{blocks}</div>;
}

// ---- reason output helpers ----
function parseBoxes(text) {
  const out = [];
  const re = /\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]/g;
  let m; while ((m = re.exec(text))) out.push([+m[1], +m[2], +m[3], +m[4]]);
  return out;
}
function parseTimes(text) {
  const out = [];
  const re = /(\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?\s*s)\s*[–\-—]+\s*(\d{1,2}:\d{2}(?:\.\d+)?|\d+(?:\.\d+)?\s*s)/gi;
  let m; while ((m = re.exec(text))) out.push(`${m[1].trim()} – ${m[2].trim()}`);
  return out;
}
function physicalVerdict(text) {
  const t = String(text).toLowerCase();
  if (/\b(implausible|not physically|physically impossible|violat|unrealistic|defies|impossible)\b/.test(t)) return { label: "Violations flagged", tone: "poor" };
  if (/\b(plausible|physically consistent|realistic|obeys|consistent with)\b/.test(t)) return { label: "Physically plausible", tone: "good" };
  return null;
}

function GroundingImage({ src, boxes }) {
  const [nat, setNat] = useState(null);
  const flat = boxes.flat();
  const max = flat.length ? Math.max(...flat) : 1;
  const pc = (v, dim) => {
    if (max <= 1) return v * 100;
    if (max <= 1000) return (v / 1000) * 100;
    return nat ? (v / nat[dim]) * 100 : (v / max) * 100;
  };
  return (
    <div className="ground-wrap">
      <img className="media" src={src} alt="input" onLoad={(e) => setNat([e.target.naturalWidth, e.target.naturalHeight])} />
      {boxes.map((b, i) => (
        <div key={i} className="ground-box"
          style={{ left: pc(b[0], 0) + "%", top: pc(b[1], 1) + "%", width: (pc(b[2], 0) - pc(b[0], 0)) + "%", height: (pc(b[3], 1) - pc(b[1], 1)) + "%" }}>
          <span className="ground-lbl">{i + 1}</span>
        </div>
      ))}
    </div>
  );
}

// Reason output: media as the subject, output coupled to it, Q→A pairing.
function ReasonResult({ mode, question, text, mediaUrl, isVideo }) {
  const id = mode?.id;
  const boxes = id === "grounding" ? parseBoxes(text) : [];
  const times = id === "temporal" ? parseTimes(text) : [];
  const verdict = id === "physical" ? physicalVerdict(text) : null;
  return (
    <div className="reasonstage">
      {mediaUrl && (
        <div className="rs-media">
          {id === "grounding" && !isVideo && boxes.length
            ? <GroundingImage src={mediaUrl} boxes={boxes} />
            : isVideo ? <video className="media" src={mediaUrl} controls loop />
              : <img className="media" src={mediaUrl} alt="input" />}
        </div>
      )}
      {times.length > 0 && (
        <div className="rs-times">{times.map((t, i) => <span key={i} className="rs-time">{t}</span>)}</div>
      )}
      <div className="rs-answer">
        {question && <div className="rs-q"><span className="rs-q-tag">Q</span><span>{question}</span></div>}
        {verdict && <div className={"rs-verdict " + verdict.tone}>{verdict.label}</div>}
        <Markdown text={text} />
      </div>
    </div>
  );
}

// Generation: show the conditioning input(s) that drove the result, then the result hero.
function InputStrip({ mode, refUrl, actionPlan, params }) {
  if (!mode || mode.reference === "none") return null;
  const isVid = mode.reference === "video";
  const ctrl = mode.id === "transfer" ? params?.control : null;
  return (
    <div className="iostrip">
      <span className="io-cap">conditioned on · {mode.reference}{ctrl ? ` · ${ctrl}` : ""}</span>
      <div className="io-row">
        {refUrl && (isVid
          ? <video className="io-thumb" src={refUrl} muted loop autoPlay />
          : <img className="io-thumb" src={refUrl} alt="input" />)}
        {actionPlan?.data?.length > 0 && (
          <div className="io-thumb io-action">
            <ActionTrajectory action={{ shape: actionPlan.shape, data: actionPlan.data, dtype: `${actionPlan.fps || ""} fps`, action_mode: "action plan", domain_id: actionPlan.domain?.domain_id }} />
          </div>
        )}
        <span className="io-arrow">▸</span>
      </div>
    </div>
  );
}

function Field({ spec, value, onChange, valueLabel }) {
  const w = spec.widget || spec.type;

  if (spec.type === "bool") {
    return (
      <label className="field toggle">
        <span>{spec.label}</span>
        <button type="button" role="switch" aria-checked={!!value}
          className={"switch" + (value ? " on" : "")} onClick={() => onChange(!value)}>
          <span className="knob" />
        </button>
      </label>
    );
  }

  if (w === "segmented") {
    const cur = String(value ?? spec.default ?? "");
    const labels = spec.optionLabels || {};
    return (
      <div className="field">
        <span className="field-label">{spec.label}</span>
        <div className="segmented">
          {spec.options.map((o) => (
            <button key={o} type="button" className={"seg-btn" + (cur === String(o) ? " active" : "")}
              onClick={() => onChange(o)}>{labels[o] || o}</button>
          ))}
        </div>
        {spec.hint && <span className="field-hint">{spec.hint}</span>}
      </div>
    );
  }

  if (w === "slider") {
    const v = value ?? spec.default ?? spec.min ?? 0;
    return (
      <div className="field">
        <span className="field-label">{spec.label}
          <span className="field-val">{valueLabel ?? `${v}${spec.unit ? " " + spec.unit : ""}`}</span>
        </span>
        <input type="range" min={spec.min} max={spec.max} step={spec.step ?? (spec.type === "int" ? 1 : "any")}
          value={v} onChange={(e) => onChange(Number(e.target.value))} />
      </div>
    );
  }

  return (
    <label className="field">
      <span className="field-label">{spec.label}</span>
      {w === "select" ? (
        <select value={value ?? spec.default ?? ""} onChange={(e) => onChange(e.target.value)}>
          {spec.options.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      ) : w === "text" ? (
        <input type="text" value={value ?? ""} placeholder="—" onChange={(e) => onChange(e.target.value)} />
      ) : (
        <input
          type="number" value={value ?? ""} min={spec.min} max={spec.max}
          step={spec.step ?? (spec.type === "int" ? 1 : "any")}
          placeholder={spec.default == null ? "random" : String(spec.default)}
          onChange={(e) => onChange(e.target.value === "" ? null : Number(e.target.value))}
        />
      )}
    </label>
  );
}

export default function App() {
  const [config, setConfig] = useState(null);
  const [modeId, setModeId] = useState(null);
  const [params, setParams] = useState({ prompt: "" });
  const [refFile, setRefFile] = useState(null);
  const [surface, setSurface] = useState("generate");
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [showMore, setShowMore] = useState(false);
  const [simple, setSimple] = useState(() => localStorage.getItem("c3_simple") !== "0");
  const [resetKey, setResetKey] = useState(0);
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);     // exact request the backend will send
  const [actionPlan, setActionPlan] = useState(null); // driving action trajectory (action modes)
  const [validation, setValidation] = useState(null); // round-trip consistency check
  const pollRef = useRef(null);
  const elapsedRef = useRef(null);
  const previewRef = useRef(null);

  useEffect(() => {
    getConfig().then((c) => { setConfig(c); setModeId(c.modes[0]?.id); }).catch((e) => setError(String(e)));
    return () => { clearTimeout(pollRef.current); clearInterval(elapsedRef.current); clearTimeout(previewRef.current); };
  }, []);

  const mode = useMemo(() => config?.modes.find((m) => m.id === modeId), [config, modeId]);
  const isReason = surface === "reason";

  // Pipeline transparency: keep a live preview of the exact request for the current settings.
  useEffect(() => {
    if (!modeId) return;
    clearTimeout(previewRef.current);
    previewRef.current = setTimeout(() => {
      requestPreview(modeId, params).then(setPreview).catch(() => setPreview(null));
    }, 300);
    return () => clearTimeout(previewRef.current);
  }, [modeId, params]);

  // Fetch the action plan that drives action modes (forward dynamics) for visualization.
  useEffect(() => {
    setActionPlan(null);
    if (mode?.action && mode?.reference === "image") {
      exampleAction(modeId).then(setActionPlan).catch(() => setActionPlan(null));
    }
  }, [modeId, mode]);

  useEffect(() => {
    if (!config || !mode) return;
    const ex = mode.example || {};
    const knobs = mode.surface === "reason" ? config.reason_knobs : config.gen_knobs;
    const next = { prompt: ex.prompt || "" };
    for (const k of knobs) if (k.default !== undefined) next[k.key] = k.default;
    for (const e of mode.extra || []) next[e.key] = e.default;
    Object.assign(next, ex.params || {});
    setParams(next);
    setRefFile(null);
    setResult(null);
    setError(null);
    if (ex.reference) {
      fetch(`/api/example/${mode.id}/reference`)
        .then((r) => (r.ok ? r.blob() : null))
        .then((b) => b && setRefFile(new File([b], ex.reference, { type: b.type })))
        .catch(() => {});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modeId, config, resetKey]);

  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));
  function pickSurface(s) {
    setSurface(s);
    const first = config?.modes.find((m) => m.surface === s && (!simple || SIMPLE_TASKS.has(m.id)));
    if (first) setModeId(first.id);
  }
  function toggleSimple() {
    const next = !simple;
    setSimple(next);
    localStorage.setItem("c3_simple", next ? "1" : "0");
    if (next && !SIMPLE_TASKS.has(modeId)) {
      const first = config?.modes.find((m) => m.surface === surface && SIMPLE_TASKS.has(m.id))
        || config?.modes.find((m) => SIMPLE_TASKS.has(m.id));
      if (first) { setSurface(first.surface); setModeId(first.id); }
    }
  }
  const refUrl = useMemo(() => (refFile ? URL.createObjectURL(refFile) : null), [refFile]);
  useEffect(() => () => refUrl && URL.revokeObjectURL(refUrl), [refUrl]);

  const knobs = useMemo(() => {
    if (!config || !mode) return [];
    const list = isReason ? config.reason_knobs : config.gen_knobs;
    // Action modes derive num_frames from the action chunk — not a free knob.
    const actionMode = !!mode.action;
    return list.filter((k) => {
      if (k.video && mode.kind !== "video") return false;
      if (actionMode && k.key === "num_frames") return false;
      return true;
    });
  }, [config, mode, isReason]);

  // Essentials = the knobs that matter for THIS task; the rest go to Advanced.
  const keyKeys = mode?.key_knobs || [];
  const hideKeys = mode?.hide_knobs || [];
  const essentialKnobs = knobs.filter((k) => keyKeys.includes(k.key) && !hideKeys.includes(k.key));
  const advancedKnobs = knobs.filter((k) => !keyKeys.includes(k.key) && !hideKeys.includes(k.key));
  const framesLabel = (k) => {
    const f = Number(params.num_frames) || 0, fp = Number(params.fps) || 24;
    return `${f} frames · ~${(f / fp).toFixed(1)}s`;
  };

  const reasonerOff = isReason && !config?.reasoner?.available;

  async function onGenerate() {
    if (!mode) return;
    setError(null); setResult(null); setValidation(null); setStatus("running");
    try {
      // policy runs autoregressively: the model predicts its own actions each chunk.
      if (mode.id === "policy") { await rolloutFlow(); return; }
      const res = await generate(mode.id, params, refFile);
      if (res.kind === "text") { setResult({ kind: "text", text: res.text }); setStatus("done"); }
      else if (res.kind === "image") { setResult({ kind: "image", src: res.src }); setStatus("done"); }
      else pollJob(res.job_id, res.async_action);
    } catch (e) { setError(String(e.message || e)); setStatus("error"); }
  }

  async function rolloutFlow() {
    const { rollout_id, total } = await rolloutStart(mode.id, params, refFile);
    const startTs = performance.now();
    setResult({ kind: "video", rolloutId: rollout_id, rolloutTotal: total, rolloutChunk: 0, jobStatus: "running", elapsed: 0 });
    clearInterval(elapsedRef.current);
    elapsedRef.current = setInterval(() => {
      setResult((r) => (r && r.rolloutId ? { ...r, elapsed: (performance.now() - startTs) / 1000 } : r));
    }, 300);
    const tick = async () => {
      try {
        const s = await rolloutStatus(rollout_id);
        setResult((r) => ({ ...r, rolloutChunk: s.chunk, jobStatus: s.status }));
        if (s.status === "completed") {
          clearInterval(elapsedRef.current);
          setResult((r) => ({ ...r, kind: "video", src: rolloutContentUrl(rollout_id), jobStatus: "completed" }));
          setStatus("done"); return;
        }
        if (s.status === "error") {
          clearInterval(elapsedRef.current); setError(s.error || "rollout failed"); setStatus("error"); return;
        }
        pollRef.current = setTimeout(tick, 1500);
      } catch (e) { clearInterval(elapsedRef.current); setError(String(e.message || e)); setStatus("error"); }
    };
    tick();
  }

  async function onValidate() {
    if (!mode || !result?.jobId) return;
    setValidation({ status: "running", domain: "agibotworld" });
    try {
      const data = await validate(mode.id, params, result.jobId);
      setValidation({ status: "done", data });
    } catch (e) {
      setValidation({ status: "error", error: String(e.message || e) });
    }
  }

  function pollJob(jobId, asyncAction) {
    const units = (Number(params.num_inference_steps) || 35) * (Number(params.num_frames) || 93);
    const secPerUnit = Number(localStorage.getItem("c3_spu")) || 0.04;
    const estTotal = Math.max(6, secPerUnit * units);
    const startTs = performance.now();
    const spec = mode?.id === "transfer"
      ? `auto-aligned to control clip · ${params.num_inference_steps || "?"} steps`
      : `${(params.size || "?").replace("x", "×")} · ${params.num_frames || "?"}f · ${params.num_inference_steps || "?"} steps`;
    setResult({ kind: "video", jobId, jobStatus: "queued", elapsed: 0, estTotal, serverProgress: 0, spec, polls: 0, asyncAction });
    clearInterval(elapsedRef.current);
    elapsedRef.current = setInterval(() => {
      setResult((r) => (r && r.kind === "video" ? { ...r, elapsed: (performance.now() - startTs) / 1000 } : r));
    }, 300);
    const stop = () => clearInterval(elapsedRef.current);
    const tick = async () => {
      try {
        const j = await getJob(jobId);
        setResult((r) => ({ ...r, jobStatus: j.status, serverProgress: j.progress || 0, polls: (r.polls || 0) + 1 }));
        if (j.status === "completed") {
          stop();
          const actual = j.inference_time_s || (performance.now() - startTs) / 1000;
          if (units > 0 && actual > 0) localStorage.setItem("c3_spu", String(actual / units));
          setResult((r) => ({ ...r, src: jobContentUrl(jobId), action: j.action, profiling: j, jobStatus: "completed" }));
          setStatus("done");
          return;
        }
        if (j.status === "failed") {
          stop();
          const e = j.error;
          setError((e && (e.message || e.detail)) || (typeof e === "string" ? e : "generation failed"));
          setStatus("error");
          return;
        }
        pollRef.current = setTimeout(tick, 1500);
      } catch (e) { stop(); setError(String(e.message || e)); setStatus("error"); }
    };
    tick();
  }

  if (error && !config) return <div className="fatal">Cannot reach playground backend: {error}</div>;
  if (!config) return <div className="loading">Loading…</div>;

  const surfaceModes = config.modes.filter((m) => m.surface === surface && (!simple || SIMPLE_TASKS.has(m.id)));
  // Generate (Advanced) is organized by downstream scenario: every task lives under
  // its scenario group, all groups shown at once (no primary-flat row, no collapse).
  const scenarioNav = surface === "generate" && !simple;
  const primaryModes = scenarioNav ? [] : surfaceModes.filter((m) => m.primary);
  const moreModes = simple ? [] : (scenarioNav ? surfaceModes : surfaceModes.filter((m) => !m.primary));
  const moreGroups = [...new Set(moreModes.map((m) => m.group))];
  const moreOpen = scenarioNav ? true : (showMore || moreModes.some((m) => m.id === modeId));
  // forward dynamics derives its frame count from the rollout selection (chunk_size · n + 1).
  const frames = mode?.id === "fwd_dynamics"
    ? (mode.chunk_size || 16) * (Number(params.rollout_chunks) || 1) + 1
    : params.num_frames;
  const specLine = isReason
    ? `${params.max_tokens || 512} tokens · temp ${params.temperature ?? 0.2}`
    : mode?.id === "transfer"
      ? `auto-aligned to control clip · ${params.num_inference_steps} steps`
      : [(params.size || "").replace("x", "×"), mode?.kind === "video" ? `${frames}f` : null,
         mode?.kind === "video" ? `${params.fps}fps` : null, `${params.num_inference_steps} steps`]
          .filter(Boolean).join(" · ");

  return (
    <div className="app" data-surface={surface}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><IconAtom /></span>
          <span><b>Cosmos3</b> <span className="brand-sub">Playground</span></span>
        </div>
        <div className="meta">
          <div className="mode-seg" role="tablist" aria-label="View mode">
            <button role="tab" aria-selected={simple} className={"mseg" + (simple ? " active" : "")}
              onClick={() => { if (!simple) toggleSimple(); }}>Simple</button>
            <button role="tab" aria-selected={!simple} className={"mseg" + (!simple ? " active" : "")}
              onClick={() => { if (simple) toggleSimple(); }}>Advanced</button>
          </div>
          <span className="pill"><span className="dot" /> {config.generator.model || "no model"}</span>
          <span className={"pill" + (config.reasoner.available ? "" : " off")}>
            <span className="dot" /> {config.reasoner.available ? config.reasoner.model || "reasoner" : "reasoner off"}
          </span>
        </div>
      </header>

      {/* workspace — viewport (canvas, left) + inspector (right: nav + controls) */}
      <div className="workspace">
        <main className="viewport">
          {error && <div className="error"><IconAlert /><span>{error}</span></div>}
          {!result && !error && status !== "running" && <div className="placeholder">{isReason ? "Attach media, ask a question, then press Analyze. The answer appears here." : "Describe a world, then press Generate. The result appears here."}</div>}

          {/* image/reason have no job to poll — show an indeterminate working state so it never looks frozen */}
          {status === "running" && (!result || result.kind !== "video") && (
            <div className="progress-wrap">
              <div className="progress-bar indeterminate"><div /></div>
              <div className="progress-text">{isReason ? "Analyzing…" : "Rendering image…"}</div>
            </div>
          )}

          {result?.rolloutId && result.jobStatus !== "completed" && status === "running" && (
            <div className="progress-wrap">
              <div className="progress-bar"><div style={{ width: `${(result.rolloutChunk / Math.max(1, result.rolloutTotal)) * 100}%` }} /></div>
              <div className="progress-text">Autoregressive rollout · chunk {result.rolloutChunk}/{result.rolloutTotal} · {fmt(result.elapsed)} elapsed</div>
              <div className="debug">forward dynamics — each chunk conditioned on the previous last frame</div>
            </div>
          )}

          {result?.kind === "video" && !result.rolloutId && result.jobStatus !== "completed" && status === "running" && (() => {
            const disp = result.serverProgress > 0 ? result.serverProgress : Math.min(96, ((result.elapsed || 0) / (result.estTotal || 1)) * 100);
            const eta = Math.max(0, (result.estTotal || 0) - (result.elapsed || 0));
            const phase = { queued: "Queued", in_progress: "Generating", running: "Generating" }[result.jobStatus] || result.jobStatus;
            return (
              <div className="progress-wrap">
                <div className="progress-bar"><div style={{ width: `${disp}%` }} /></div>
                <div className="progress-text">{phase} · {Math.round(disp)}% · {fmt(result.elapsed)} elapsed{result.serverProgress > 0 ? " (server)" : ` · ~${fmt(eta)} left`}</div>
                <div className="debug">{result.spec} · job #{(result.jobId || "").slice(-8)}</div>
              </div>
            );
          })()}

          {(() => {
            // Inverse dynamics' real output is the action trajectory; the server also returns a
            // reconstruction clip, but showing it reads as "it just echoed my video". Surface the
            // trajectory instead and skip the video for action-output modes.
            const actionOut = result?.action?.action_mode === "inverse_dynamics";
            return (<>
              {result?.kind === "image" && result.src && <img className="media" src={result.src} alt="result" />}
              {result?.kind === "video" && result.src && !actionOut && <video className="media" src={result.src} controls autoPlay loop />}
              {result?.kind === "text" && <div className="answer"><Markdown text={result.text} /></div>}
              {actionOut && result?.action && <ActionTrajectory action={result.action} />}
              {result?.action && !actionOut && (
                <div className="action-box"><b>action</b> · mode={result.action.action_mode} · shape={JSON.stringify(result.action.shape)} · dtype={result.action.dtype} · domain={result.action.domain_id}</div>
              )}
              {result?.profiling?.inference_time_s != null && (
                <div className="profiling">
                  {result.profiling.inference_time_s?.toFixed?.(1)}s
                  {result.profiling.peak_memory_mb ? ` · ${Math.round(result.profiling.peak_memory_mb)} MB peak` : ""}
                  {result.src && result.kind === "video" && !actionOut ? <> · <a href={result.src} download><IconDownload /> mp4</a></> : ""}
                </div>
              )}
              {mode?.id === "fwd_dynamics" && result?.kind === "video" && result.jobStatus === "completed" && result.jobId && (
                <ValidationCard state={validation} onRun={onValidate} />
              )}
            </>);
          })()}
        </main>

        <aside className="inspector">
          {/* surface + task picker live at the top of the right control panel */}
          <div className="inspector-nav">
            <div className="surface-seg" role="tablist" aria-label="Surface">
              <button role="tab" aria-selected={surface === "generate"}
                className={"seg" + (surface === "generate" ? " active" : "")}
                onClick={() => pickSurface("generate")}><IconSparkles /> Generate</button>
              <button role="tab" aria-selected={surface === "reason"}
                className={"seg" + (surface === "reason" ? " active" : "")}
                onClick={() => pickSurface("reason")}><IconEye /> Reason</button>
            </div>
            <div className="tabs">
              <div className="tab-row">
                {primaryModes.map((m) => (
                  <button key={m.id} className={"tab" + (m.id === modeId ? " active" : "")}
                    onClick={() => setModeId(m.id)} title={m.blurb}>{m.label}</button>
                ))}
                {moreModes.length > 0 && !scenarioNav && (
                  <button type="button" className="more-toggle" aria-expanded={moreOpen}
                    onClick={() => setShowMore((v) => !v)}>
                    <span className={"chev" + (moreOpen ? " open" : "")}><IconChevron /></span>
                    More tasks
                  </button>
                )}
              </div>
              {moreOpen && moreGroups.map((g) => (
                <div key={g} className="tab-group">
                  <div className="tab-group-label">{g}</div>
                  <div className="tab-row">
                    {moreModes.filter((m) => m.group === g).map((m) => (
                      <button key={m.id} className={"tab" + (m.id === modeId ? " active" : "")}
                        onClick={() => setModeId(m.id)} title={m.blurb}>{m.label}</button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="task-head">
            <div className="task-head-text">
              <h2 className="task-title">{mode?.label}</h2>
              {mode?.purpose ? (
                <div className="purpose">
                  <p className="purpose-text">{mode.purpose}</p>
                  {mode.flow && (
                    <div className="flow" aria-label="inputs to output">
                      {mode.flow.inputs.flatMap((c, i) => [
                        i > 0 && <span key={"p" + i} className="flow-plus">+</span>,
                        <span key={"c" + i} className="flow-chip in">{c}</span>,
                      ])}
                      <span className="flow-arrow">→</span>
                      <span className="flow-chip out">{mode.flow.output}</span>
                    </div>
                  )}
                  {mode.notes && (
                    <ul className="notes">
                      {mode.notes.map(([k, v], i) => (
                        <li key={i}>
                          <span className={"note-key " + (i === 0 ? "keep" : "chg")}>{k}</span>
                          <span className="note-val">{v}</span>
                        </li>
                      ))}
                    </ul>
                  )}
                </div>
              ) : (
                <>
                  {mode?.io && <div className="io-line">{mode.io}</div>}
                  <p className="blurb">{mode?.blurb}</p>
                </>
              )}
            </div>
            <button type="button" className="reset-btn" title="Restore the official example prompt + settings"
              onClick={() => setResetKey((k) => k + 1)}><IconReset /> Reset</button>
          </div>
          {mode?.note && <div className="warn-inline"><IconAlert /><span>{mode.note}</span></div>}

          <label className="field">
            <span className="field-label">{isReason ? "Question" : "Prompt"}</span>
            <textarea rows={3} value={params.prompt || ""}
              placeholder={isReason ? "Ask a question about the media…" : "Describe the world to render…"}
              onChange={(e) => setParam("prompt", e.target.value)} />
          </label>

          {mode && mode.reference !== "none" && (
            <label className="field dropzone">
              <span className="field-label"><IconUpload /> {isReason ? "Media" : "Reference"} ({mode.reference})</span>
              <input type="file" accept={mode.reference === "image" ? "image/*" : "video/*"}
                onChange={(e) => setRefFile(e.target.files?.[0] || null)} />
              {refFile && <span className="ref-name">{refFile.name}</span>}
              {refUrl && (mode.reference === "image"
                ? <img className="ref-preview" src={refUrl} alt="reference" />
                : <video className="ref-preview" src={refUrl} controls muted loop autoPlay playsInline />)}
            </label>
          )}

          {!simple && (mode?.extra?.length > 0 || essentialKnobs.length > 0) && (
            <div className="essentials">
              {(mode.extra || []).map((e) => (
                <Field key={e.key} spec={e} value={params[e.key]} onChange={(v) => {
                  // each control type keeps its OWN strength (remembered per type)
                  if (e.key === "control" && mode.control_defaults?.[v] != null) {
                    setParams((p) => ({ ...p, control: v,
                      control_guidance: (p.cg_by_type?.[v] ?? mode.control_defaults[v]) }));
                  } else if (e.key === "control_guidance") {
                    setParams((p) => ({ ...p, control_guidance: v,
                      cg_by_type: { ...(p.cg_by_type || {}), [p.control || "edge"]: v } }));
                  } else { setParam(e.key, v); }
                }} />
              ))}
              {essentialKnobs.map((k) => (
                <Field key={k.key} spec={k} value={params[k.key]} onChange={(v) => setParam(k.key, v)}
                  valueLabel={k.key === "num_frames" ? framesLabel(k) : undefined} />
              ))}
            </div>
          )}

          {!simple && advancedKnobs.length > 0 && (
            <div className="section">
              <button type="button" className="disclosure" aria-expanded={showAdvanced}
                onClick={() => setShowAdvanced((v) => !v)}>
                <span className={"chev" + (showAdvanced ? " open" : "")}><IconChevron /></span>
                Advanced
                <span className="disclosure-hint">{specLine}</span>
              </button>
              {showAdvanced && (
                <div className="grid" style={{ marginTop: 10 }}>
                  {advancedKnobs.map((k) => <Field key={k.key} spec={k} value={params[k.key]} onChange={(v) => setParam(k.key, v)} />)}
                </div>
              )}
            </div>
          )}

          {reasonerOff && <div className="warn-inline"><IconAlert /><span>Reasoner not connected — start the playground with <code>--reasoner-url</code>.</span></div>}
          <div className="commit-bar">
            <button className="generate" disabled={status === "running" || reasonerOff} onClick={onGenerate}>
              {status !== "running" && (isReason ? <IconEye /> : <IconSparkles />)}
              {status === "running" ? (isReason ? "Analyzing…" : "Generating…") : isReason ? "Analyze" : "Generate"}
            </button>
          </div>
        </aside>
      </div>

      {/* bottom console dock — pipeline transparency (Advanced only) */}
      {!simple && (
      <section className="dock">
        {(
          <div className="dock-body">
            {preview?.domain && (
              <div className="domain-card">
                <span className="domain-name">{preview.domain.name}</span>
                <span className="domain-kind">{preview.domain.kind}</span>
                <span className="domain-meta">
                  {preview.domain.raw_action_dim ? `${preview.domain.raw_action_dim}-dim` : "dim n/a"}
                  {preview.domain.fps ? ` · ${preview.domain.fps} fps` : ""}
                  {preview.domain.viewpoint ? ` · ${preview.domain.viewpoint}` : ""}
                  {` · domain_id ${preview.domain.domain_id}`}
                </span>
              </div>
            )}
            {preview?.graph?.nodes?.length > 0 && (
              <ExecutionTopology topology={config.generator?.topology} graph={preview.graph} />
            )}
            <div className="dock-cols">
              {preview?.surface === "generate" && (
                <div className="pipe-stage">
                  <pre className="req-json">{JSON.stringify({ fields: preview.fields, extra_params: preview.extra_params }, null, 2)}</pre>
                </div>
              )}
              {preview?.surface === "reason" && (
                <div className="pipe-stage">
                  <pre className="req-json">{JSON.stringify({ prompt: preview.prompt, max_tokens: preview.max_tokens, temperature: preview.temperature }, null, 2)}</pre>
                </div>
              )}
              {actionPlan?.data?.length > 0 && (
                <div className="pipe-stage">
                  <div className="pipe-label">Action plan — drives the simulation ({actionPlan.num_chunks} chunks × {actionPlan.action_chunk_size} steps)</div>
                  <ActionTrajectory title="Action plan (input — drives the generated video)" action={{
                    shape: actionPlan.shape, data: actionPlan.data,
                    dtype: actionPlan.fps ? `${actionPlan.fps} fps` : "input",
                    action_mode: "input plan", domain_id: actionPlan.domain?.domain_id,
                  }} />
                </div>
              )}
              {!preview && <div className="pipe-empty">building request preview…</div>}
            </div>
          </div>
        )}
      </section>
      )}
    </div>
  );
}
