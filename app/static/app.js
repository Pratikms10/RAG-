const $ = (id) => document.getElementById(id);
const state = { documents: [] };

function node(tag, className, text) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== undefined) element.textContent = text;
  return element;
}

function fileKind(name) {
  return name.toLowerCase().endsWith(".pdf") ? "PDF" : "TXT";
}

function score(value) {
  return value === null || value === undefined ? "—" : Number(value).toFixed(2);
}

async function request(path, options = {}) {
  const response = await fetch(path, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || "The local service could not complete that request.");
  return payload;
}

function setConnection(health) {
  const connection = document.querySelector(".connection");
  const label = $("connectionLabel");
  connection.classList.add("ready");
  label.textContent = `${health.documents} document${health.documents === 1 ? "" : "s"} · ${health.chunks} chunks`;
}

function renderDocuments(documents) {
  state.documents = documents;
  const list = $("documentList");
  list.replaceChildren();
  $("documentCount").textContent = `${documents.length} doc${documents.length === 1 ? "" : "s"}`;
  $("clearButton").disabled = documents.length === 0;
  for (const document of documents) {
    const fragment = $("documentTemplate").content.cloneNode(true);
    fragment.querySelector(".file-tag").textContent = fileKind(document.filename);
    fragment.querySelector(".document-name").textContent = document.filename;
    fragment.querySelector(".document-meta").textContent = `${document.chunk_count} chunk${document.chunk_count === 1 ? "" : "s"} · ${document.chunking_strategy}`;
    list.append(fragment);
  }
}

async function refreshIndex() {
  const [health, documents] = await Promise.all([request("/health"), request("/documents")]);
  setConnection(health);
  renderDocuments(documents);
}

function showError(message) {
  const shell = $("resultShell");
  shell.dataset.state = "error";
  const card = node("div", "error-card");
  card.append(node("p", "eyebrow", "LOCAL SERVICE RESPONSE"), node("h2", "", "The evidence trace could not run."), node("p", "", message));
  shell.replaceChildren(card);
}

function terms(items) {
  const list = node("div", "term-list");
  if (!items.length) return list;
  for (const item of items) list.append(node("span", "term", item));
  return list;
}

function evidenceSpan(span) {
  const card = node("article", "span-card");
  const footer = node("footer");
  footer.append(node("span", "", span.chunk_id), node("span", "", `${Math.round(span.coverage * 100)}% of meaningful terms`));
  card.append(node("p", "", `“${span.text}”`), terms(span.matched_terms), footer);
  return card;
}

function candidateCard(candidate, index) {
  const details = node("details", "candidate");
  const summary = node("summary");
  const top = node("div", "candidate-top");
  const text = node("div");
  const fileLine = candidate.page_number ? `${candidate.filename} · page ${candidate.page_number}` : candidate.filename;
  text.append(node("b", "", `Candidate ${index + 1}`), node("small", "", fileLine));
  top.append(node("span", "rank", String(index + 1)), text, node("span", "candidate-score", score(candidate.score)));
  summary.append(top);
  const body = node("div", "candidate-body");
  body.append(node("p", "", candidate.excerpt));
  const signals = node("div", "signal-line");
  signals.append(node("span", "", `dense ${score(candidate.dense_score)}`), node("span", "", `lexical ${score(candidate.lexical_score)}`), node("span", "", `hybrid ${score(candidate.score)}`));
  body.append(signals, terms(candidate.matched_terms));
  details.append(summary, body);
  return details;
}

function renderResult(result) {
  const trace = result.retrieval_trace;
  const shell = $("resultShell");
  shell.dataset.state = result.grounded ? "grounded" : "insufficient";
  const root = node("article", "result");
  const head = node("div", "result-head");
  head.append(node("span", `verdict ${result.grounded ? "yes" : "no"}`, result.grounded ? "GROUNDED" : "INSUFFICIENT EVIDENCE"), node("span", "backend-label", `answer mode · ${result.answer_backend}`));
  root.append(head, node("p", "answer-copy", result.answer));
  for (const warning of result.warnings) root.append(node("p", "warning", warning));

  const summary = node("div", "trace-summary");
  const metricValues = [
    ["Top hybrid score", `${score(trace.top_score)} / ${score(trace.minimum_score)}`],
    ["Term coverage", `${Math.round(trace.term_coverage * 100)}%`],
    ["Single-span check", trace.requires_single_span ? `${Math.round(trace.best_span_coverage * 100)}%` : "multi-fact allowed"],
  ];
  for (const [label, value] of metricValues) {
    const metric = node("div", "trace-metric");
    metric.append(node("small", "", label), node("b", "", value));
    summary.append(metric);
  }
  root.append(summary);

  const why = node("section", "why-panel");
  why.append(node("h3", "", result.grounded ? "Why this answer was allowed" : "Why the gate held"));
  const reasons = node("ul");
  for (const reason of trace.reasons) reasons.append(node("li", "", reason));
  why.append(reasons);
  root.append(why);

  if (trace.evidence_spans.length) {
    const section = node("section", "evidence-section");
    const heading = node("h3", "evidence-heading");
    heading.append(node("span", "", "DIRECT SENTENCE SUPPORT"), node("span", "", `${trace.evidence_spans.length} span${trace.evidence_spans.length === 1 ? "" : "s"}`));
    section.append(heading);
    trace.evidence_spans.slice(0, 3).forEach((span) => section.append(evidenceSpan(span)));
    root.append(section);
  }

  if (trace.candidates.length) {
    const section = node("section", "evidence-section");
    const heading = node("h3", "evidence-heading");
    heading.append(node("span", "", "RETRIEVAL TRACE"), node("span", "", `${trace.candidates.length} ranked chunks`));
    const grid = node("div", "candidate-grid");
    trace.candidates.forEach((candidate, index) => grid.append(candidateCard(candidate, index)));
    section.append(heading, grid);
    root.append(section);
  }
  shell.replaceChildren(root);
}

async function upload(file) {
  if (!file) return;
  const size = Number($("chunkSize").value);
  const overlap = Number($("chunkOverlap").value);
  if (!Number.isInteger(size) || !Number.isInteger(overlap) || overlap >= size) {
    showError("Chunk overlap must be a whole number smaller than chunk size.");
    return;
  }
  const data = new FormData();
  data.append("file", file);
  data.append("chunk_size_words", String(size));
  data.append("chunk_overlap_words", String(overlap));
  const dropzone = $("dropzone");
  dropzone.classList.add("dragging");
  try {
    await request("/documents", { method: "POST", body: data });
    await refreshIndex();
  } catch (error) {
    showError(error.message);
  } finally {
    dropzone.classList.remove("dragging");
    $("fileInput").value = "";
  }
}

async function ask(event) {
  event.preventDefault();
  const question = $("questionInput").value.trim();
  if (question.length < 3) return;
  const button = $("askButton");
  button.disabled = true;
  button.querySelector("span").textContent = "Tracing…";
  try {
    const result = await request("/questions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: Number($("topK").value) }),
    });
    renderResult(result);
  } catch (error) {
    showError(error.message);
  } finally {
    button.disabled = false;
    button.querySelector("span").textContent = "Trace answer";
  }
}

function wireEvents() {
  $("fileInput").addEventListener("change", (event) => upload(event.target.files[0]));
  const dropzone = $("dropzone");
  ["dragenter", "dragover"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.add("dragging"); }));
  ["dragleave", "drop"].forEach((name) => dropzone.addEventListener(name, (event) => { event.preventDefault(); dropzone.classList.remove("dragging"); }));
  dropzone.addEventListener("drop", (event) => upload(event.dataTransfer.files[0]));
  $("questionForm").addEventListener("submit", ask);
  document.querySelectorAll("[data-prompt]").forEach((button) => button.addEventListener("click", () => { $("questionInput").value = button.dataset.prompt; $("questionInput").focus(); }));
  $("clearButton").addEventListener("click", async () => {
    if (!window.confirm("Clear every locally indexed document?")) return;
    try { await request("/documents", { method: "DELETE" }); await refreshIndex(); } catch (error) { showError(error.message); }
  });
}

wireEvents();
refreshIndex().catch((error) => showError(error.message));
