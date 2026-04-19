/* Shipment Classifier — CSV tester
 *
 * Stateless single-page app. Lifecycle:
 *   1. User drops a CSV → PapaParse → validate headers → POST /classify/batch
 *      with persist:false on every row.
 *   2. Render results into a table. No localStorage / sessionStorage / cookies.
 *
 * Risk is reported per-shipment via `compliance.is_risky` (true/false).
 * No block/review tiers and no per-category risk tiers.
 */

(() => {
  "use strict";

  // ── Config ────────────────────────────────────────────────────────────────

  const REQUIRED_COLUMNS = ["shipment_id", "cargo_description", "commodity_description"];
  const OPTIONAL_COLUMNS = ["threshold", "unclassified_threshold"];
  const MAX_ROWS         = 500;   // matches /classify/batch server-side cap
  const TRUNC_LEN        = 80;    // chars shown in cargo/commodity columns

  // 20 distinct category colours (Tableau 20 qualitative palette).
  // Keys are the exact category names defined in classification_categories.
  // Alphabetical mapping — adding a 21st category appends a colour without
  // reshuffling existing assignments.
  const CATEGORY_COLOURS = {
    agriculture:     "#4E79A7",
    automotive:      "#A0CBE8",
    chemicals:       "#F28E2B",
    construction:    "#FFBE7D",
    cosmetics:       "#59A14F",
    defense:         "#8CD17D",
    electronics:     "#B6992D",
    energy:          "#F1CE63",
    food_beverages:  "#499894",
    furniture:       "#86BCB6",
    luxury:          "#E15759",
    machinery:       "#FF9D9A",
    metals:          "#79706E",
    minerals:        "#BAB0AC",
    paper:           "#D37295",
    perishables:     "#FABFD2",
    pharmaceuticals: "#B07AA1",
    plastics:        "#D4A6C8",
    textiles:        "#9D7660",
    toys:            "#D7B5A6",
  };
  const FALLBACK_COLOUR = "#cbd2d9";

  // ── DOM helpers ───────────────────────────────────────────────────────────

  const $ = (sel) => document.querySelector(sel);
  const dropzone   = $("#dropzone");
  const fileInput  = $("#file");
  const pickBtn    = $("#pickBtn");
  const banner     = $("#banner");
  const status     = $("#status");
  const statusText = $("#statusText");
  const results    = $("#results");
  const summary    = $("#summary");
  const tbody      = $("#resultsBody");

  // ── Boot ──────────────────────────────────────────────────────────────────

  pickBtn.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => {
    const file = e.target.files && e.target.files[0];
    if (file) handleFile(file);
  });

  ["dragenter", "dragover"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.add("dragover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropzone.classList.remove("dragover");
    })
  );
  dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files && e.dataTransfer.files[0];
    if (file) handleFile(file);
  });
  dropzone.addEventListener("click", (e) => {
    // Click anywhere in the dropzone (except on the explicit button)
    if (e.target.tagName !== "BUTTON" && e.target.tagName !== "A") {
      fileInput.click();
    }
  });

  // ── Pipeline ──────────────────────────────────────────────────────────────

  function handleFile(file) {
    hideBanner();
    hideResults();
    showStatus(`Parsing ${file.name}…`);

    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (h) => (h || "").trim(),
      complete: (parsed) => onParsed(parsed),
      error: (err) => {
        hideStatus();
        showBanner(`CSV parse error: ${err.message || err}`, "error");
      },
    });
  }

  function onParsed(parsed) {
    const rows = parsed.data || [];
    const headers = parsed.meta && parsed.meta.fields ? parsed.meta.fields : [];

    const missing = REQUIRED_COLUMNS.filter((c) => !headers.includes(c));
    if (missing.length) {
      hideStatus();
      showBanner(
        `CSV is missing required column(s): ${missing.map((m) => `"${m}"`).join(", ")}. ` +
        `Required: ${REQUIRED_COLUMNS.join(", ")}. Optional: ${OPTIONAL_COLUMNS.join(", ")}.`,
        "error"
      );
      return;
    }

    if (!rows.length) {
      hideStatus();
      showBanner("CSV had headers but no data rows.", "warn");
      return;
    }

    let workingRows = rows;
    let warning = null;
    if (rows.length > MAX_ROWS) {
      workingRows = rows.slice(0, MAX_ROWS);
      warning = `CSV had ${rows.length} rows; processing the first ${MAX_ROWS} (server-side batch limit).`;
    }

    const shipments = workingRows.map((r) => buildShipment(r));
    classifyBatch(shipments, warning);
  }

  function buildShipment(row) {
    const ship = {
      shipment_id:           String(row.shipment_id || "").trim(),
      cargo_description:     String(row.cargo_description || "").trim(),
      commodity_description: String(row.commodity_description || "").trim(),
      persist:               false,   // hard guarantee of statelessness
    };
    if (row.threshold !== undefined && row.threshold !== "") {
      const t = Number(row.threshold);
      if (!Number.isNaN(t)) ship.threshold = t;
    }
    if (row.unclassified_threshold !== undefined && row.unclassified_threshold !== "") {
      const u = Number(row.unclassified_threshold);
      if (!Number.isNaN(u)) ship.unclassified_threshold = u;
    }
    return ship;
  }

  async function classifyBatch(shipments, warning) {
    showStatus(`Classifying ${shipments.length} shipment${shipments.length === 1 ? "" : "s"}…`);
    try {
      const t0 = performance.now();
      const resp = await fetch("/classify/batch", {
        method:  "POST",
        headers: { "Content-Type": "application/json" },
        body:    JSON.stringify({ shipments }),
      });
      if (!resp.ok) {
        const text = await resp.text();
        throw new Error(`HTTP ${resp.status}: ${text.slice(0, 300)}`);
      }
      const data = await resp.json();
      const elapsed = ((performance.now() - t0) / 1000).toFixed(2);
      hideStatus();
      if (warning) showBanner(warning, "warn");
      renderResults(data, elapsed);
    } catch (err) {
      hideStatus();
      showBanner(`Classification failed: ${err.message || err}`, "error");
    }
  }

  // ── Rendering ─────────────────────────────────────────────────────────────

  function renderResults(rows, elapsed) {
    tbody.innerHTML = "";

    let classifiedCount   = 0;
    let lowConfCount      = 0;
    let unclassifiedCount = 0;
    let riskyCount        = 0;
    let mismatchCount     = 0;
    let rerankedCount     = 0;

    for (const row of rows) {
      const tr = document.createElement("tr");
      const result     = row.result     || {};
      const compl      = row.compliance || {};
      const meta       = row.meta       || {};
      const input      = row.input      || {};
      const conf       = result.confidence_state || "unclassified";
      const matched    = result.categories || [];
      const scores     = result.scores || {};
      const isRisky    = compl.is_risky === true;
      const reasons    = compl.decision_reasons || [];
      const hsCodes    = result.hs_codes_extracted || [];
      const hsImplied  = result.hs_implied_categories || [];
      const hsMismatch = result.hs_text_mismatch === true;

      if (conf === "classified")     classifiedCount++;
      else if (conf === "low_confidence") lowConfCount++;
      else                            unclassifiedCount++;
      if (isRisky) {
        riskyCount++;
        tr.classList.add("risky-row");
      }
      if (hsMismatch) {
        mismatchCount++;
        tr.classList.add("hs-mismatch-row");
      }
      if (meta.reranked === true) rerankedCount++;

      tr.appendChild(td("cell-id",     row.shipment_id || ""));
      tr.appendChild(td("cell-text",   truncCell(input.cargo_description)));
      tr.appendChild(td("cell-text",   truncCell(input.commodity_description)));
      tr.appendChild(labelsCell(matched, scores, conf));
      tr.appendChild(hsSignalCell(hsCodes, hsImplied, hsMismatch));
      tr.appendChild(confidenceCell(conf));
      tr.appendChild(complianceCell(isRisky, reasons));
      tr.appendChild(td("cell-chunks", meta.chunks_processed && meta.chunks_processed > 1
                                       ? String(meta.chunks_processed) : ""));

      tbody.appendChild(tr);
    }

    summary.textContent =
      `${rows.length} shipment${rows.length === 1 ? "" : "s"} in ${elapsed}s — ` +
      `${classifiedCount} classified, ${lowConfCount} low_confidence, ${unclassifiedCount} unclassified · ` +
      `${riskyCount} flagged risky · ${mismatchCount} HS\u2194text mismatch${mismatchCount === 1 ? "" : "es"}` +
      (rerankedCount > 0 ? ` · ${rerankedCount} reranked` : "");
    results.classList.remove("hidden");
  }

  function td(className, text) {
    const cell = document.createElement("td");
    if (className) cell.className = className;
    if (text instanceof Node) cell.appendChild(text);
    else cell.textContent = text || "";
    return cell;
  }

  function truncCell(text) {
    text = text || "";
    const span = document.createElement("span");
    span.className = "truncate";
    span.textContent = text.length > TRUNC_LEN ? text.slice(0, TRUNC_LEN) + "…" : text;
    span.title = text;   // full text on hover
    return span;
  }

  function labelsCell(matched, scores, conf) {
    const cell = document.createElement("td");
    cell.className = "cell-labels";

    if (matched && matched.length) {
      for (const cat of matched) {
        cell.appendChild(categoryChip(cat, scores[cat]));
      }
      return cell;
    }

    // Unclassified: grey "unclassified" chip + up to 3 grey runner-up chips.
    const grey = document.createElement("span");
    grey.className = "chip unclassified";
    grey.textContent = "unclassified";
    cell.appendChild(grey);

    const runners = Object.entries(scores)
      .map(([cat, s]) => [cat, (s && s.final_score) || 0])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3);
    for (const [cat, score] of runners) {
      const chip = document.createElement("span");
      chip.className = "chip runner-up";
      chip.textContent = `considered: ${cat} (${score.toFixed(2)})`;
      cell.appendChild(chip);
    }
    return cell;
  }

  function hsSignalCell(codes, implied, mismatch) {
    const cell = document.createElement("td");
    cell.className = "cell-hs";

    if (!codes || !codes.length) {
      const dash = document.createElement("span");
      dash.className = "hs-none";
      dash.textContent = "\u2014";
      cell.appendChild(dash);
      return cell;
    }

    for (const code of codes) {
      const chip = document.createElement("span");
      chip.className = "chip hs-code";
      chip.textContent = code;
      cell.appendChild(chip);
    }

    if (implied && implied.length) {
      const arrow = document.createElement("span");
      arrow.className = "hs-arrow";
      arrow.textContent = "\u2192";
      cell.appendChild(arrow);
      for (const cat of implied) {
        cell.appendChild(categoryChip(cat, null));
      }
    }

    if (mismatch) {
      cell.classList.add("mismatch");
      const badge = document.createElement("span");
      badge.className = "badge mismatch";
      badge.textContent = "HS \u2260 text";
      cell.appendChild(badge);
    }

    return cell;
  }

  function categoryChip(cat, scoreObj) {
    const chip = document.createElement("span");
    chip.className = "chip";
    const bg = CATEGORY_COLOURS[cat] || FALLBACK_COLOUR;
    chip.style.background = bg;
    chip.style.color = pickTextColour(bg);

    const name = document.createElement("span");
    name.textContent = cat;
    chip.appendChild(name);

    if (scoreObj && typeof scoreObj.final_score === "number") {
      chip.title = `final_score=${scoreObj.final_score.toFixed(3)}`;
    }
    return chip;
  }

  function confidenceCell(conf) {
    const cell = document.createElement("td");
    cell.className = "cell-conf";
    const span = document.createElement("span");
    span.className = `confidence ${conf}`;
    span.textContent = conf;
    cell.appendChild(span);
    return cell;
  }

  function complianceCell(isRisky, reasons) {
    const cell = document.createElement("td");
    cell.className = "cell-compl";
    const badge = document.createElement("span");
    badge.className = `badge ${isRisky ? "risky" : "clean"}`;
    badge.textContent = isRisky ? "RISKY" : "clean";
    cell.appendChild(badge);
    if (isRisky && reasons && reasons.length) {
      const r = document.createElement("span");
      r.className = "compl-reasons";
      r.textContent = reasons.slice(0, 2).join(" · ");
      r.title = reasons.join("\n");
      cell.appendChild(r);
    }
    return cell;
  }

  // WCAG-AA-ish text colour pick. Returns black on light backgrounds, white on dark.
  function pickTextColour(hex) {
    const h = hex.replace("#", "");
    const r = parseInt(h.slice(0, 2), 16) / 255;
    const g = parseInt(h.slice(2, 4), 16) / 255;
    const b = parseInt(h.slice(4, 6), 16) / 255;
    // Relative luminance per WCAG.
    const lin = (c) => (c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));
    const L = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
    return L > 0.5 ? "#1a1f2b" : "#ffffff";
  }

  // ── UI state helpers ─────────────────────────────────────────────────────

  function showStatus(msg) {
    statusText.textContent = msg;
    status.classList.remove("hidden");
  }
  function hideStatus()  { status.classList.add("hidden"); }
  function hideResults() { results.classList.add("hidden"); }
  function hideBanner()  { banner.classList.add("hidden"); banner.className = "banner hidden"; }
  function showBanner(msg, kind) {
    banner.textContent = msg;
    banner.className = "banner " + (kind === "warn" ? "warn" : kind === "info" ? "info" : "");
    banner.classList.remove("hidden");
  }
})();
