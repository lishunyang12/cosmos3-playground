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
