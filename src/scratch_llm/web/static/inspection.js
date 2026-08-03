const JSON_HEADERS = Object.freeze({
  accept: "application/json",
  "content-type": "application/json",
});


async function responseError(response, fallback) {
  try {
    const payload = await response.json();
    if (typeof payload?.error?.message === "string") {
      return new Error(payload.error.message);
    }
  } catch (_error) {
    // The stable fallback is safer than reflecting an arbitrary response body.
  }
  return new Error(fallback);
}


export async function requestJson(fetchImpl, path, options = {}) {
  const response = await fetchImpl(path, {
    ...options,
    headers: {...JSON_HEADERS, ...options.headers},
  });
  if (!response.ok) {
    throw await responseError(response, "The local server request failed.");
  }
  return response.json();
}


async function acknowledgedMutation(fetchImpl, path, body, commit) {
  const payload = await requestJson(fetchImpl, path, {
    method: "POST",
    body: JSON.stringify(body),
  });
  commit(payload);
  return payload;
}


export function loadCheckpoint(fetchImpl, checkpointId, commit) {
  return acknowledgedMutation(
    fetchImpl,
    "/api/load_checkpoint",
    {checkpoint_id: checkpointId},
    commit,
  );
}


export function selectRenderer(fetchImpl, rendererId, commit) {
  return acknowledgedMutation(
    fetchImpl,
    "/api/select_renderer",
    {renderer_id: rendererId},
    commit,
  );
}


export function populateSelect(
  select,
  entries,
  activeId,
  documentRef = document,
) {
  select.replaceChildren();
  for (const entry of entries) {
    const option = documentRef.createElement("option");
    option.value = entry.id;
    option.textContent = entry.name;
    select.appendChild(option);
  }
  if (entries.some((entry) => entry.id === activeId)) {
    select.value = activeId;
  }
  select.disabled = entries.length === 0;
}


function finite(value) {
  return typeof value === "number" && Number.isFinite(value);
}


export function formatMetrics(metrics) {
  return {
    generatedTokens: Number.isInteger(metrics?.generated_tokens)
      ? String(metrics.generated_tokens)
      : "—",
    prefill: finite(metrics?.prefill_latency_seconds)
      ? `${(1000 * metrics.prefill_latency_seconds).toFixed(1)} ms`
      : "—",
    decode: finite(metrics?.decode_latency_per_sampled_token_seconds)
      ? `${(1000 * metrics.decode_latency_per_sampled_token_seconds).toFixed(1)} ms/token`
      : "—",
    throughput: finite(metrics?.tokens_per_second)
      ? `${metrics.tokens_per_second.toFixed(2)} tok/s`
      : "—",
    memory: finite(metrics?.peak_memory_mib)
      ? `${metrics.peak_memory_mib.toFixed(1)} MiB`
      : "—",
  };
}


export function renderMetrics(elements, metrics) {
  const labels = formatMetrics(metrics);
  elements.generatedTokens.textContent = labels.generatedTokens;
  elements.prefill.textContent = labels.prefill;
  elements.decode.textContent = labels.decode;
  elements.throughput.textContent = labels.throughput;
  elements.memory.textContent = labels.memory;
}


export function renderDebug(element, debug, state) {
  element.textContent = debug
    ? JSON.stringify(
        {
          ...debug,
          context: state?.context ?? null,
        },
        null,
        2,
      )
    : "Enable raw token debug before sending a message.";
}


function downloadFilename(header) {
  const match = /filename="([A-Za-z0-9._-]+)"/.exec(header ?? "");
  return match?.[1] ?? "scratch-llm-transcript.jsonl";
}


export async function downloadTranscript(
  fetchImpl,
  documentRef,
  urlApi = URL,
) {
  const response = await fetchImpl("/api/transcript", {
    headers: {accept: "application/x-ndjson"},
  });
  if (!response.ok) {
    throw await responseError(
      response,
      "A completed conversation is required for export.",
    );
  }
  const objectUrl = urlApi.createObjectURL(await response.blob());
  const anchor = documentRef.createElement("a");
  anchor.href = objectUrl;
  anchor.download = downloadFilename(response.headers.get("content-disposition"));
  documentRef.body.appendChild(anchor);
  try {
    anchor.click();
  } finally {
    anchor.remove();
    urlApi.revokeObjectURL(objectUrl);
  }
}
