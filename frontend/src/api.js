// Tiny API wrapper around the playground backend.

export async function getConfig() {
  const r = await fetch("/api/config");
  if (!r.ok) throw new Error("failed to load config");
  return r.json();
}

export async function generate(mode, params, referenceFile) {
  const form = new FormData();
  form.append("mode", mode);
  form.append("params", JSON.stringify(params));
  if (referenceFile) form.append("reference", referenceFile);
  const r = await fetch("/api/generate", { method: "POST", body: form });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `generate failed (${r.status})`);
  }
  // Images come back as raw binary (smaller + progressive paint); everything else is JSON.
  const ct = r.headers.get("content-type") || "";
  if (ct.startsWith("image/")) {
    return { kind: "image", src: URL.createObjectURL(await r.blob()) };
  }
  return r.json();
}

export async function getJob(jobId) {
  const r = await fetch(`/api/jobs/${jobId}`);
  if (!r.ok) throw new Error("failed to poll job");
  return r.json();
}

export function jobContentUrl(jobId) {
  return `/api/jobs/${jobId}/content`;
}

// Pipeline transparency: the exact request the backend will send, for current settings.
export async function requestPreview(mode, params) {
  const r = await fetch("/api/request-preview", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, params }),
  });
  if (!r.ok) throw new Error("preview failed");
  return r.json();
}

// Autoregressive forward-dynamics rollout: generate chunk-by-chunk on the server,
// each conditioned on the previous chunk's last frame, then stitch the clips.
export async function rolloutStart(mode, params, referenceFile) {
  const form = new FormData();
  form.append("mode", mode);
  form.append("params", JSON.stringify(params));
  if (referenceFile) form.append("reference", referenceFile);
  const r = await fetch("/api/rollout", { method: "POST", body: form });
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(d.detail || `rollout failed (${r.status})`);
  }
  return r.json(); // { rollout_id, total }
}
export async function rolloutStatus(id) {
  const r = await fetch(`/api/rollout/${id}`);
  if (!r.ok) throw new Error("rollout status failed");
  return r.json();
}
export function rolloutContentUrl(id) {
  return `/api/rollout/${id}/content`;
}

// Round-trip validation: re-read the action out of a forward-dynamics video with
// inverse dynamics (same domain) and score it against the original plan.
export async function validate(mode, params, jobId) {
  const r = await fetch("/api/validate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mode, params, job_id: jobId }),
  });
  if (!r.ok) {
    const detail = await r.json().catch(() => ({}));
    throw new Error(detail.detail || `validation failed (${r.status})`);
  }
  return r.json();
}

// The action plan that drives an action mode (forward dynamics), for visualization.
export async function exampleAction(mode) {
  const r = await fetch(`/api/example/${mode}/action`);
  if (!r.ok) return null;
  return r.json();
}

// Pre-baked example output (gallery default): the saved result for a mode, shown until
// the user regenerates. Returns null when nothing has been baked for this mode.
export async function exampleResult(mode) {
  const r = await fetch(`/api/example/${mode}/result`);
  if (!r.ok) return null;
  return r.json();
}
export function exampleResultContentUrl(mode) {
  return `/api/example/${mode}/result/content`;
}
