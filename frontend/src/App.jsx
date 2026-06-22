import { useEffect, useMemo, useRef, useState } from "react";
import { getConfig, generate, getJob, jobContentUrl } from "./api.js";

function fmt(sec) {
  if (sec == null || isNaN(sec)) return "0s";
  sec = Math.max(0, Math.round(sec));
  return sec < 60 ? `${sec}s` : `${Math.floor(sec / 60)}m${String(sec % 60).padStart(2, "0")}s`;
}

function Field({ spec, value, onChange }) {
  const t = spec.type;
  if (t === "bool") {
    return (
      <label className="field checkbox">
        <input type="checkbox" checked={!!value} onChange={(e) => onChange(e.target.checked)} />
        <span>{spec.label}</span>
      </label>
    );
  }
  return (
    <label className="field">
      <span className="field-label">{spec.label}</span>
      {t === "select" ? (
        <select value={value ?? spec.default ?? ""} onChange={(e) => onChange(e.target.value)}>
          {spec.options.map((o) => (
            <option key={o} value={o}>{o}</option>
          ))}
        </select>
      ) : t === "text" ? (
        <input type="text" value={value ?? ""} placeholder="—" onChange={(e) => onChange(e.target.value)} />
      ) : (
        <input
          type="number"
          value={value ?? ""}
          min={spec.min}
          max={spec.max}
          step={spec.step ?? (t === "int" ? 1 : "any")}
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
  const [status, setStatus] = useState("idle"); // idle|running|done|error
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const pollRef = useRef(null);
  const elapsedRef = useRef(null);

  useEffect(() => {
    getConfig()
      .then((c) => {
        setConfig(c);
        setModeId(c.modes[0]?.id);
      })
      .catch((e) => setError(String(e)));
    return () => {
      clearTimeout(pollRef.current);
      clearInterval(elapsedRef.current);
    };
  }, []);

  const mode = useMemo(() => config?.modes.find((m) => m.id === modeId), [config, modeId]);

  // reset knob defaults when mode changes
  useEffect(() => {
    if (!config || !mode) return;
    const next = { prompt: params.prompt || "" };
    for (const k of config.knobs) if (k.default !== undefined) next[k.key] = k.default;
    for (const e of mode.extra || []) next[e.key] = e.default;
    setParams(next);
    setRefFile(null);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [modeId, config]);

  const setParam = (k, v) => setParams((p) => ({ ...p, [k]: v }));

  const visibleKnobs = useMemo(() => {
    if (!config || !mode) return [];
    return config.knobs.filter((k) => !(k.video && mode.kind !== "video"));
  }, [config, mode]);

  async function onGenerate() {
    if (!mode) return;
    setError(null);
    setResult(null);
    setStatus("running");
    try {
      const res = await generate(mode.id, params, refFile);
      if (res.kind === "image") {
        setResult({ kind: "image", src: `data:image/${res.format};base64,${res.b64}` });
        setStatus("done");
      } else {
        pollJob(res.job_id);
      }
    } catch (e) {
      setError(String(e.message || e));
      setStatus("error");
    }
  }

  function pollJob(jobId) {
    // Cosmos reports progress as 0 then 100, so estimate from elapsed time vs a
    // self-calibrating per-unit cost (units = steps x frames), refined each run
    // from the server's reported inference_time_s.
    const units = (Number(params.num_inference_steps) || 35) * (Number(params.num_frames) || 93);
    const secPerUnit = Number(localStorage.getItem("c3_spu")) || 0.04;
    const estTotal = Math.max(6, secPerUnit * units);
    const startTs = performance.now();

    setResult({ kind: "video", jobId, jobStatus: "queued", elapsed: 0, estTotal, serverProgress: 0 });
    clearInterval(elapsedRef.current);
    elapsedRef.current = setInterval(() => {
      setResult((r) => (r && r.kind === "video" ? { ...r, elapsed: (performance.now() - startTs) / 1000 } : r));
    }, 300);

    const stop = () => clearInterval(elapsedRef.current);
    const tick = async () => {
      try {
        const job = await getJob(jobId);
        setResult((r) => ({ ...r, jobStatus: job.status, serverProgress: job.progress || 0 }));
        if (job.status === "completed") {
          stop();
          const actual = job.inference_time_s || (performance.now() - startTs) / 1000;
          if (units > 0 && actual > 0) localStorage.setItem("c3_spu", String(actual / units));
          setResult((r) => ({ ...r, src: jobContentUrl(jobId), action: job.action, profiling: job, jobStatus: "completed" }));
          setStatus("done");
          return;
        }
        if (job.status === "failed") {
          stop();
          const e = job.error;
          setError((e && (e.message || e.detail)) || (typeof e === "string" ? e : "generation failed"));
          setStatus("error");
          return;
        }
        pollRef.current = setTimeout(tick, 1500);
      } catch (e) {
        stop();
        setError(String(e.message || e));
        setStatus("error");
      }
    };
    tick();
  }

  if (error && !config) return <div className="fatal">Cannot reach playground backend: {error}</div>;
  if (!config) return <div className="loading">Loading…</div>;

  const groups = [...new Set(config.modes.map((m) => m.group))];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">⚛ <b>Cosmos3</b> Playground</div>
        <div className="meta">
          <span className="pill">{config.model || "model?"}</span>
          <span className="muted">{config.server_url}</span>
        </div>
      </header>

      <div className="layout">
        <aside className="controls">
          <div className="tabs">
            {groups.map((g) => (
              <div key={g} className="tab-group">
                <div className="tab-group-label">{g}</div>
                <div className="tab-row">
                  {config.modes.filter((m) => m.group === g).map((m) => (
                    <button
                      key={m.id}
                      className={"tab" + (m.id === modeId ? " active" : "")}
                      onClick={() => setModeId(m.id)}
                      title={m.blurb}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>

          <p className="blurb">{mode?.blurb}</p>

          <label className="field">
            <span className="field-label">Prompt</span>
            <textarea
              rows={3}
              value={params.prompt || ""}
              placeholder="A cinematic drone shot over a misty forest at dawn…"
              onChange={(e) => setParam("prompt", e.target.value)}
            />
          </label>

          {mode && mode.reference !== "none" && (
            <label className="field dropzone">
              <span className="field-label">Reference ({mode.reference})</span>
              <input
                type="file"
                accept={mode.reference === "image" ? "image/*" : "video/*"}
                onChange={(e) => setRefFile(e.target.files?.[0] || null)}
              />
              {refFile && <span className="ref-name">{refFile.name}</span>}
            </label>
          )}

          {mode?.extra?.length > 0 && (
            <div className="section">
              <div className="section-title">{mode.label} options</div>
              <div className="grid">
                {mode.extra.map((e) => (
                  <Field key={e.key} spec={e} value={params[e.key]} onChange={(v) => setParam(e.key, v)} />
                ))}
              </div>
            </div>
          )}

          <div className="section">
            <div className="section-title">Settings</div>
            <div className="grid">
              {visibleKnobs.map((k) => (
                <Field key={k.key} spec={k} value={params[k.key]} onChange={(v) => setParam(k.key, v)} />
              ))}
            </div>
          </div>

          <button className="generate" disabled={status === "running"} onClick={onGenerate}>
            {status === "running" ? "Generating…" : "Generate"}
          </button>
        </aside>

        <main className="stage">
          {error && <div className="error">⚠ {error}</div>}
          {!result && !error && <div className="placeholder">Pick a mode, write a prompt, hit Generate.</div>}

          {result?.kind === "video" && result.jobStatus !== "completed" && status === "running" && (() => {
            const disp = result.serverProgress > 0
              ? result.serverProgress
              : Math.min(96, ((result.elapsed || 0) / (result.estTotal || 1)) * 100);
            const eta = Math.max(0, (result.estTotal || 0) - (result.elapsed || 0));
            return (
              <div className="progress-wrap">
                <div className="progress-bar"><div style={{ width: `${disp}%` }} /></div>
                <div className="progress-text">
                  {result.jobStatus} · {Math.round(disp)}% · {fmt(result.elapsed)} elapsed
                  {result.serverProgress > 0 ? "" : ` · ~${fmt(eta)} left`}
                </div>
              </div>
            );
          })()}

          {result?.kind === "image" && result.src && <img className="media" src={result.src} alt="result" />}
          {result?.kind === "video" && result.src && (
            <video className="media" src={result.src} controls autoPlay loop />
          )}

          {result?.action && (
            <div className="action-box">
              <b>action</b> · mode={result.action.action_mode} · shape={JSON.stringify(result.action.shape)} · dtype={result.action.dtype}
            </div>
          )}
          {result?.profiling?.inference_time_s != null && (
            <div className="profiling">
              {result.profiling.inference_time_s?.toFixed?.(1)}s
              {result.profiling.peak_memory_mb ? ` · ${Math.round(result.profiling.peak_memory_mb)} MB peak` : ""}
              {result.src && result.kind === "video" ? <> · <a href={result.src} download>download mp4</a></> : ""}
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
