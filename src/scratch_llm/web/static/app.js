import {
  downloadTranscript,
  loadCheckpoint,
  populateSelect,
  renderDebug,
  renderMetrics,
  requestJson,
  selectRenderer,
} from "./inspection.js";


const API_VERSION = "v1";


function numericValue(control, {integer, optional = false}, label) {
  const raw = control.value;
  if (typeof raw !== "string" || raw.trim() === "") {
    if (optional) {
      return null;
    }
    throw new Error(`Invalid generation settings: ${label} is required.`);
  }
  const value = Number(raw);
  const min = Number(control.min);
  const max = Number(control.max);
  if (
    !Number.isFinite(value) ||
    !Number.isFinite(min) ||
    !Number.isFinite(max) ||
    (integer && !Number.isInteger(value)) ||
    value < min ||
    value > max
  ) {
    throw new Error(`Invalid generation settings: ${label} is out of range.`);
  }
  return value;
}


export function normalizeSettings(values) {
  return {
    temperature: numericValue(
      values.temperature,
      {integer: false},
      "temperature",
    ),
    top_k: numericValue(
      values.topK,
      {integer: true, optional: true},
      "top k",
    ),
    max_new_tokens: numericValue(
      values.maxNewTokens,
      {integer: true},
      "max new tokens",
    ),
  };
}


export function buildGenerateMessage(message, settings, debug = false) {
  return {
    protocol_version: API_VERSION,
    type: "generate",
    message,
    debug,
    settings,
  };
}


export function appendSafeText(element, text, documentRef = document) {
  if (typeof text !== "string" || text === "") {
    return;
  }
  element.appendChild(documentRef.createTextNode(text));
}


function countLabel(value, singular, plural) {
  return `${value} ${value === 1 ? singular : plural}`;
}


export function contextLabel(state) {
  const context = state?.context;
  if (!context) {
    return "No checkpoint context available";
  }
  const details = [`${context.prompt_tokens} / ${context.max_tokens} tokens`];
  if (context.dropped_turns > 0) {
    details.push(
      `${countLabel(context.dropped_turns, "older turn", "older turns")} dropped`,
    );
  }
  if (context.truncated_user_tokens > 0) {
    details.push(
      `${countLabel(context.truncated_user_tokens, "user token", "user tokens")} truncated`,
    );
  }
  return details.join(" · ");
}


function disable(control, value) {
  if (control) {
    control.disabled = value;
  }
}


export function applyControlState(
  controls,
  state,
  {
    sessionReady = true,
    transcriptAvailable = false,
    hasCheckpoints = true,
    hasRenderers = true,
  } = {},
) {
  const active = ["connecting", "generating", "stopping"].includes(state);
  const locked = state !== "idle";
  disable(controls.input, locked || !sessionReady);
  disable(controls.send, locked || !sessionReady);
  disable(controls.stop, !active || state === "stopping");
  disable(controls.reset, locked || !sessionReady);
  disable(controls.temperature, locked || !sessionReady);
  disable(controls.topK, locked || !sessionReady);
  disable(controls.maxNewTokens, locked || !sessionReady);
  disable(controls.checkpoint, locked || !hasCheckpoints);
  disable(controls.loadCheckpoint, locked || !hasCheckpoints);
  disable(controls.renderer, locked || !sessionReady || !hasRenderers);
  disable(controls.applyRenderer, locked || !sessionReady || !hasRenderers);
  disable(controls.exportTranscript, locked || !transcriptAvailable);
  disable(controls.debug, locked || !sessionReady);
}


export function websocketUrl(locationRef) {
  const scheme = locationRef.protocol === "https:" ? "wss:" : "ws:";
  return `${scheme}//${locationRef.host}/ws/generate`;
}


export function applyServerEvent(payload, view) {
  if (!payload || payload.protocol_version !== API_VERSION) {
    throw new Error("Unsupported server protocol response.");
  }
  if (payload.type === "start") {
    view.start();
    return false;
  }
  if (payload.type === "token") {
    const text = payload.event?.text_delta;
    if (typeof text === "string" && text !== "") {
      view.appendAssistant(text);
    }
    return false;
  }
  if (payload.type === "done") {
    const text = payload.event?.text_delta;
    if (typeof text === "string" && text !== "") {
      view.appendAssistant(text);
    }
    view.finish("done", payload.state, payload);
    return true;
  }
  if (["cancelled", "busy", "error"].includes(payload.type)) {
    view.finish(payload.type, payload.state ?? null, payload);
    return true;
  }
  throw new Error("Unknown server event.");
}


export async function resetConversation(fetchImpl, view) {
  const payload = await requestJson(fetchImpl, "/api/reset", {
    method: "POST",
  });
  view.clearConversation();
  view.updateContext(payload.state);
  return payload.state;
}


function messageNode(documentRef, role, text) {
  const article = documentRef.createElement("article");
  const heading = documentRef.createElement("h2");
  const body = documentRef.createElement("p");
  article.className = `message ${role}`;
  heading.textContent = role === "user" ? "You" : "Assistant";
  appendSafeText(body, text, documentRef);
  article.append(heading, body);
  return {article, body};
}


export class ChatController {
  constructor({documentRef, fetchImpl, WebSocketImpl, locationRef}) {
    this.document = documentRef;
    this.fetch = fetchImpl;
    this.WebSocket = WebSocketImpl;
    this.location = locationRef;
    this.form = documentRef.getElementById("chat-form");
    this.log = documentRef.getElementById("chat-log");
    this.status = documentRef.getElementById("connection-status");
    this.context = documentRef.getElementById("context-status");
    this.metrics = {
      generatedTokens: documentRef.getElementById("generated-token-metric"),
      prefill: documentRef.getElementById("prefill-metric"),
      decode: documentRef.getElementById("decode-metric"),
      throughput: documentRef.getElementById("throughput-metric"),
      memory: documentRef.getElementById("memory-metric"),
    };
    this.debugOutput = documentRef.getElementById("debug-output");
    this.controls = {
      input: documentRef.getElementById("message-input"),
      temperature: documentRef.getElementById("temperature"),
      topK: documentRef.getElementById("top-k"),
      maxNewTokens: documentRef.getElementById("max-new-tokens"),
      send: documentRef.getElementById("send-button"),
      stop: documentRef.getElementById("stop-button"),
      reset: documentRef.getElementById("reset-button"),
      checkpoint: documentRef.getElementById("checkpoint-select"),
      loadCheckpoint: documentRef.getElementById("load-checkpoint-button"),
      renderer: documentRef.getElementById("renderer-select"),
      applyRenderer: documentRef.getElementById("apply-renderer-button"),
      exportTranscript: documentRef.getElementById("export-button"),
      debug: documentRef.getElementById("debug-enabled"),
    };
    this.state = "idle";
    this.sessionReady = false;
    this.transcriptAvailable = false;
    this.hasCheckpoints = false;
    this.hasRenderers = false;
    this.socket = null;
    this.terminalReceived = false;
    this.stopSent = false;
    this.activeUser = null;
    this.activeAssistant = null;
    this.activeAssistantBody = null;
    this.bind();
    this.transition("idle");
  }

  bind() {
    this.form.addEventListener("submit", (event) => {
      event.preventDefault();
      this.submit();
    });
    this.controls.input.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && !event.shiftKey) {
        event.preventDefault();
        this.form.requestSubmit();
      }
    });
    this.controls.stop.addEventListener("click", () => this.stop());
    this.controls.reset.addEventListener("click", () => this.reset());
    this.controls.loadCheckpoint.addEventListener("click", () => {
      this.loadSelectedCheckpoint();
    });
    this.controls.applyRenderer.addEventListener("click", () => {
      this.applySelectedRenderer();
    });
    this.controls.exportTranscript.addEventListener("click", () => {
      this.exportTranscript();
    });
  }

  transition(state) {
    this.state = state;
    applyControlState(this.controls, state, {
      sessionReady: this.sessionReady,
      transcriptAvailable: this.transcriptAvailable,
      hasCheckpoints: this.hasCheckpoints,
      hasRenderers: this.hasRenderers,
    });
  }

  async initialize() {
    this.transition("loading");
    this.setStatus("Discovering local checkpoints…");
    try {
      const [checkpointCatalog, rendererCatalog, session] = await Promise.all([
        requestJson(this.fetch, "/api/checkpoints"),
        requestJson(this.fetch, "/api/renderers"),
        requestJson(this.fetch, "/api/session"),
      ]);
      this.hasCheckpoints = checkpointCatalog.checkpoints.length > 0;
      this.hasRenderers = rendererCatalog.renderers.length > 0;
      populateSelect(
        this.controls.checkpoint,
        checkpointCatalog.checkpoints,
        checkpointCatalog.active_checkpoint_id,
        this.document,
      );
      populateSelect(
        this.controls.renderer,
        rendererCatalog.renderers,
        rendererCatalog.active_renderer_id,
        this.document,
      );
      this.sessionReady = session.state.status === "ready";
      this.updateContext(session.state);
      this.setStatus(
        this.sessionReady ? "Checkpoint ready" : "Load a checkpoint to begin.",
      );
    } catch (error) {
      this.setStatus(error.message);
    } finally {
      this.transition("idle");
      this.controls.input.focus();
    }
  }

  setStatus(text) {
    this.status.textContent = text;
  }

  updateContext(state) {
    this.context.textContent = contextLabel(state);
  }

  addMessage(role, text) {
    const node = messageNode(this.document, role, text);
    this.log.appendChild(node.article);
    this.log.scrollTop = this.log.scrollHeight;
    return node;
  }

  submit() {
    if (this.state !== "idle") {
      return;
    }
    if (!this.sessionReady) {
      this.setStatus("Load a checkpoint first.");
      return;
    }
    const message = this.controls.input.value;
    if (message.trim() === "") {
      this.setStatus("Enter a message first.");
      return;
    }
    let settings;
    try {
      settings = normalizeSettings(this.controls);
    } catch (error) {
      this.setStatus(error.message);
      return;
    }

    const user = this.addMessage("user", message);
    const assistant = this.addMessage("assistant", "");
    this.activeUser = user.article;
    this.activeAssistant = assistant.article;
    this.activeAssistantBody = assistant.body;
    this.terminalReceived = false;
    this.stopSent = false;
    this.transition("connecting");
    this.setStatus("Connecting…");

    const socket = new this.WebSocket(websocketUrl(this.location));
    this.socket = socket;
    socket.addEventListener("open", () => {
      if (this.socket !== socket) {
        return;
      }
      socket.send(
        JSON.stringify(
          buildGenerateMessage(message, settings, this.controls.debug.checked),
        ),
      );
      this.controls.input.value = "";
      this.setStatus("Waiting for the server…");
    });
    socket.addEventListener("message", (event) => {
      if (this.socket !== socket) {
        return;
      }
      try {
        const payload = JSON.parse(event.data);
        applyServerEvent(payload, {
          start: () => {
            this.transition("generating");
            this.setStatus("Generating…");
          },
          appendAssistant: (text) => {
            appendSafeText(this.activeAssistantBody, text, this.document);
            this.log.scrollTop = this.log.scrollHeight;
          },
          finish: (kind, state, terminal) => this.finish(kind, state, terminal),
        });
      } catch (_error) {
        this.finish("error", null, {
          error: {message: "The server sent an invalid response."},
        });
      }
    });
    socket.addEventListener("error", () => {
      if (!this.terminalReceived) {
        this.setStatus("Connection error; this turn was not saved.");
      }
    });
    socket.addEventListener("close", () => {
      if (this.socket !== socket) {
        return;
      }
      this.socket = null;
      if (!this.terminalReceived) {
        this.removeActiveTurn();
        this.transition("idle");
        this.setStatus("Disconnected; submit again to reconnect.");
        this.controls.input.focus();
      }
    });
  }

  stop() {
    if (!["connecting", "generating"].includes(this.state) || this.stopSent) {
      return;
    }
    this.stopSent = true;
    this.transition("stopping");
    this.setStatus("Stopping…");
    if (this.socket?.readyState === this.WebSocket.OPEN) {
      this.socket.send(
        JSON.stringify({protocol_version: API_VERSION, type: "stop"}),
      );
    } else {
      this.socket?.close();
    }
  }

  finish(kind, state, terminal) {
    this.terminalReceived = true;
    if (kind === "done") {
      this.updateContext(state);
      renderMetrics(this.metrics, terminal.metrics);
      renderDebug(
        this.debugOutput,
        this.controls.debug.checked ? terminal.debug : null,
        state,
      );
      this.transcriptAvailable = true;
      this.clearActiveTurn();
      this.setStatus("Complete");
    } else {
      this.removeActiveTurn();
      this.setStatus(
        kind === "cancelled"
          ? "Generation stopped; this turn was not saved."
          : terminal?.error?.message ?? "Generation failed.",
      );
    }
    this.transition("idle");
    this.socket?.close();
    this.controls.input.focus();
  }

  removeActiveTurn() {
    this.activeUser?.remove();
    this.activeAssistant?.remove();
    this.clearActiveTurn();
  }

  clearActiveTurn() {
    this.activeUser = null;
    this.activeAssistant = null;
    this.activeAssistantBody = null;
  }

  clearConversation() {
    this.log.replaceChildren();
    this.clearActiveTurn();
    this.transcriptAvailable = false;
    renderMetrics(this.metrics, null);
    renderDebug(this.debugOutput, null, null);
  }

  async performOperation({
    state,
    pendingStatus,
    successStatus,
    operation,
    onError = () => {},
  }) {
    this.transition(state);
    this.setStatus(pendingStatus);
    try {
      await operation();
      this.setStatus(successStatus);
    } catch (error) {
      onError();
      this.setStatus(error.message);
    } finally {
      this.transition("idle");
      this.controls.input.focus();
    }
  }

  async reset() {
    if (this.state !== "idle") {
      return;
    }
    return this.performOperation({
      state: "resetting",
      pendingStatus: "Resetting…",
      successStatus: "Conversation reset.",
      operation: () => resetConversation(this.fetch, {
        clearConversation: () => this.clearConversation(),
        updateContext: (state) => this.updateContext(state),
      }),
    });
  }

  async loadSelectedCheckpoint() {
    if (this.state !== "idle" || !this.hasCheckpoints) {
      return;
    }
    const priorReady = this.sessionReady;
    return this.performOperation({
      state: "loading",
      pendingStatus: "Loading checkpoint…",
      successStatus: "Checkpoint loaded.",
      operation: () => loadCheckpoint(
        this.fetch,
        this.controls.checkpoint.value,
        (payload) => {
          this.clearConversation();
          this.sessionReady = payload.state.status === "ready";
          this.updateContext(payload.state);
        },
      ),
      onError: () => {
        this.sessionReady = priorReady;
      },
    });
  }

  async applySelectedRenderer() {
    if (this.state !== "idle" || !this.sessionReady || !this.hasRenderers) {
      return;
    }
    return this.performOperation({
      state: "switching",
      pendingStatus: "Applying prompt template…",
      successStatus: "Prompt template ready.",
      operation: () => selectRenderer(
        this.fetch,
        this.controls.renderer.value,
        (payload) => {
          if (payload.history_reset) {
            this.clearConversation();
          }
          this.updateContext(payload.state);
        },
      ),
    });
  }

  async exportTranscript() {
    if (this.state !== "idle" || !this.transcriptAvailable) {
      return;
    }
    return this.performOperation({
      state: "exporting",
      pendingStatus: "Preparing transcript…",
      successStatus: "Transcript downloaded.",
      operation: () => downloadTranscript(this.fetch, this.document),
    });
  }
}


if (typeof document !== "undefined" && typeof window !== "undefined") {
  window.addEventListener("DOMContentLoaded", () => {
    const controller = new ChatController({
      documentRef: document,
      fetchImpl: window.fetch.bind(window),
      WebSocketImpl: window.WebSocket,
      locationRef: window.location,
    });
    controller.initialize();
  });
}
