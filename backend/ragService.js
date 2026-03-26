const path = require("path");
const { spawn } = require("child_process");
const readline = require("readline");
const http = require("http");
const https = require("https");
const config = require("./config");
const { getDb } = require("./db");
let embedWorker;
let embedRl;
let workerReady = false;
let reqSeq = 0;
const pending = new Map();
const NO_DATA_TEXT = "Data not available, please try asking again";
const EMBED_WORKER_STDERR_MAX = 8;
const DEFAULT_AREA_RADIUS_KM = 3.5;
let embedWorkerLastStderr = [];

function captureEmbedWorkerStderr(line) {
  if (!line) return;
  embedWorkerLastStderr.push(line);
  if (embedWorkerLastStderr.length > EMBED_WORKER_STDERR_MAX) {
    embedWorkerLastStderr = embedWorkerLastStderr.slice(-EMBED_WORKER_STDERR_MAX);
  }
}

function formatEmbedWorkerDebug() {
  const stderrSummary = embedWorkerLastStderr.length
    ? embedWorkerLastStderr.join(" | ")
    : "no-stderr-captured";
  return `pythonCommand=${config.pythonCommand} script=${config.embeddingScriptPath} timeoutMs=${config.embedTimeoutMs} lastStderr=${stderrSummary}`;
}

function startEmbedWorker() {
  if (embedWorker && !embedWorker.killed) return;

  const scriptPath = path.resolve(__dirname, config.embeddingScriptPath);
  console.log(
    `[RAG] Starting embed worker | python=${config.pythonCommand} | script=${scriptPath} | timeoutMs=${config.embedTimeoutMs}`
  );
  embedWorkerLastStderr = [];
  embedWorker = spawn(config.pythonCommand, [scriptPath], {
    cwd: path.resolve(__dirname, ".."),
    stdio: ["pipe", "pipe", "pipe"],
  });

  embedRl = readline.createInterface({ input: embedWorker.stdout });
  workerReady = false;

  embedRl.on("line", (line) => {
    try {
      const msg = JSON.parse(line);
      if (msg.type === "ready") {
        workerReady = true;
        console.log(
          `[RAG] Embedding worker ready | model=${msg.model} | device=${msg.device}`
        );
        return;
      }
      const req = pending.get(msg.id);
      if (!req) return;
      clearTimeout(req.timer);
      pending.delete(msg.id);
      if (msg.error) req.reject(new Error(`Embedding worker error: ${msg.error}`));
      else req.resolve(msg.embedding);
    } catch (err) {
      console.error("[RAG] Failed parsing embed worker output:", err.message);
    }
  });

  embedWorker.stderr.on("data", (chunk) => {
    const line = chunk.toString().trim();
    captureEmbedWorkerStderr(line);
    console.error("[RAG][embed-worker][stderr]", line);
  });

  embedWorker.on("error", (err) => {
    console.error("[RAG] Embedding worker failed to start:", err.message);
    workerReady = false;
    for (const [, req] of pending) {
      clearTimeout(req.timer);
      req.reject(
        new Error(
          `Embedding worker process error: ${err.message}. ${formatEmbedWorkerDebug()}`
        )
      );
    }
    pending.clear();
  });

  embedWorker.on("close", (code) => {
    console.error(`[RAG] Embedding worker exited (code=${code})`);
    workerReady = false;
    for (const [, req] of pending) {
      clearTimeout(req.timer);
      req.reject(new Error("Embedding worker exited before responding"));
    }
    pending.clear();
  });
}

function buildPrompt(question, contexts) {
  const contextBlock = contexts
    .map((c, i) => {
      return `[${i + 1}] ${c.title || "Untitled"}\nURL: ${c.url || "N/A"}\n${c.text}`;
    })
    .join("\n\n");

  return [
    "You are a civic governance assistant for Karnataka policy documents.",
    "Use only the provided context. If context is insufficient, say so clearly.",
    "Provide a concise answer and include citation markers like [1], [2].",
    "",
    `Question: ${question}`,
    "",
    "Context:",
    contextBlock,
  ].join("\n");
}

function embedText(text) {
  return new Promise((resolve, reject) => {
    const start = Date.now();
    startEmbedWorker();
    const id = ++reqSeq;
    const timer = setTimeout(() => {
      pending.delete(id);
      reject(
        new Error(
          `Embedding timed out after ${config.embedTimeoutMs}ms. Check Python env/model load.`
        )
      );
    }, config.embedTimeoutMs);

    pending.set(id, {
      timer,
      resolve: (vec) => {
        console.log(
          `[RAG] Query embedding ready in ${Date.now() - start}ms (dim=${vec.length})`
        );
        resolve(vec);
      },
      reject,
    });

    const send = () => {
      const payload = JSON.stringify({ id, text });
      embedWorker.stdin.write(payload + "\n");
    };

    if (workerReady) send();
    else {
      const waitStart = Date.now();
      const interval = setInterval(() => {
        if (workerReady) {
          clearInterval(interval);
          send();
          return;
        }
        if (Date.now() - waitStart > config.embedTimeoutMs) {
          clearInterval(interval);
          clearTimeout(timer);
          pending.delete(id);
          reject(
            new Error(
              `Embedding worker startup timed out after ${config.embedTimeoutMs}ms. ${formatEmbedWorkerDebug()}`
            )
          );
        }
      }, 200);
    }
  });
}

async function retrieveRelevantChunks(question, topK = 5) {
  const start = Date.now();
  console.log("[RAG] Generating query embedding...");
  const embedding = await embedText(question);
  const db = await getDb();
  const collection = db.collection(config.mongoCollectionName);

  const pipeline = [
    {
      $vectorSearch: {
        index: config.mongoVectorIndexName,
        path: "embedding",
        queryVector: embedding,
        numCandidates: Math.max(topK * 20, 100),
        limit: topK,
      },
    },
    {
      $project: {
        _id: 0,
        chunk_id: 1,
        doc_id: 1,
        title: 1,
        url: 1,
        text: 1,
        score: { $meta: "vectorSearchScore" },
      },
    },
  ];

  const chunks = await collection.aggregate(pipeline).toArray();
  console.log(
    `[RAG] Vector search returned ${chunks.length} chunks in ${Date.now() - start}ms`
  );
  return chunks;
}

async function answerWithRag(question, topK = 5) {
  console.log("[RAG] Starting answerWithRag...");
  const chunks = await retrieveRelevantChunks(question, topK);
  const prompt = buildPrompt(question, chunks);
  console.log(
    `[RAG] Sending prompt to ${getEffectiveProvider().toUpperCase()} (chunks=${chunks.length})`
  );

  const answer = await generateTextFromPrompt(prompt, {
    temperature: 0.2,
    maxTokens: config.ollamaNumPredict,
  });
  console.log(`[RAG] Model response received (answerLen=${answer.length})`);
  const citations = chunks.map((chunk, idx) => ({
    id: idx + 1,
    chunk_id: chunk.chunk_id,
    doc_id: chunk.doc_id,
    title: chunk.title || "Untitled",
    url: chunk.url || "",
    score: Number(chunk.score || 0),
  }));

  return { answer, citations };
}

function parseGovEmpIntent(question) {
  const q = question.toLowerCase();
  const pendingWords = ["pending", "not solved", "open", "in progress","unresolved", "assigned", "investigating", "reopened"];
  const resolvedWords = ["resolved", "closed", "completed", "solved"];

  const departmentMatch = question.match(
    /department\s+(?:named|name|called)?\s*[:\-]?\s*([a-z0-9&().,\- ]+)/i
  );
  const areaMatch = question.match(
    /(area|ward|zone|panchayat|city|village|locality)\s+(?:named|name|called)?\s*[:\-]?\s*([a-z0-9&().,\- ]+)/i
  );
  const contractorMatch = question.match(
    /(contractor|vendor)\s+(?:named|name|called)?\s*[:\-]?\s*([a-z0-9&().,\- ]+)/i
  );

  const wantsDeptWise =
    q.includes("department wise") || q.includes("by department");
  const wantsAreaWise = q.includes("area wise") || q.includes("by area");

  let status = "any";
  if (pendingWords.some((w) => q.includes(w))) status = "pending";
  if (resolvedWords.some((w) => q.includes(w))) status = "resolved";

  return {
    status,
    departmentName: (departmentMatch?.[1] || "").trim(),
    areaName: (areaMatch?.[2] || "").trim(),
    contractorName: (contractorMatch?.[2] || "").trim(),
    wantsDeptWise,
    wantsAreaWise,
  };
}

function statusRegexForIntent(status) {
  if (status === "pending") {
    return /registered|open|in[-_\s]?progress|pending|assigned|investigating|reopened/i;
  }
  if (status === "resolved") {
    return /resolved|closed|completed|done|solved|fixed/i;
  }
  return null;
}

function textContains(value, search) {
  if (!value || !search) return false;
  return String(value).toLowerCase().includes(String(search).toLowerCase());
}

function normalizeMatchText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function escapeRegex(value) {
  return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function candidateNameVariants(name) {
  const normalized = normalizeMatchText(name);
  if (!normalized) return [];

  const variants = new Set([normalized]);
  const withoutDept = normalized
    .replace(/\bdepartment\b/g, " ")
    .replace(/\bdept\b/g, " ")
    .replace(/\s+/g, " ")
    .trim();
  if (withoutDept) variants.add(withoutDept);

  return Array.from(variants);
}

function findBestNamedEntityMatch(question, candidates) {
  const normalizedQuestion = normalizeMatchText(question);
  if (!normalizedQuestion || !Array.isArray(candidates) || !candidates.length) {
    return { entity: null, score: 0 };
  }

  let best = null;
  let bestScore = 0;

  for (const candidate of candidates) {
    const candidateName = String(candidate?.name || "").trim();
    if (!candidateName) continue;

    const variants = candidateNameVariants(candidateName);
    let score = 0;

    for (const variant of variants) {
      if (!variant) continue;
      const bounded = new RegExp(`(^|\\s)${escapeRegex(variant)}(\\s|$)`, "i");
      if (bounded.test(normalizedQuestion)) {
        score = Math.max(score, 100 + variant.length);
        continue;
      }
      if (normalizedQuestion.includes(variant)) {
        score = Math.max(score, 70 + variant.length);
      }
    }

    if (score > bestScore) {
      bestScore = score;
      best = candidate;
    }
  }

  return { entity: best, score: bestScore };
}

function findBestNamedEntityInQuestion(question, candidates) {
  return findBestNamedEntityMatch(question, candidates).entity;
}

function extractAreaLikePhrase(question) {
  const q = String(question || "");

  const explicit = q.match(
    /(area|ward|zone|panchayat|city|village|locality)\s+(?:named|name|called)?\s*[:\-]?\s*([a-z0-9&().,\- ]+)/i
  );
  if (explicit?.[2]) return explicit[2].trim();

  const generic = q.match(/\b(?:in|at|near)\s+([a-z0-9&().,\- ]+?)(?:\?|$)/i);
  if (!generic?.[1]) return "";
  const candidate = generic[1].trim();

  if (/\bdepartment\b|\bdept\b|\bcontractor\b|\bvendor\b/i.test(candidate)) {
    return "";
  }
  return candidate;
}

function parseLocationCoordinates(value) {
  const text = String(value || "").trim();
  if (!text) return null;

  // Supports patterns like "Location: 12.294893, 76.633146"
  const m = text.match(/(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)/);
  if (!m) return null;

  const lat = Number(m[1]);
  const lon = Number(m[2]);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

function toRadians(deg) {
  return (Number(deg || 0) * Math.PI) / 180;
}

function haversineKm(lat1, lon1, lat2, lon2) {
  const R = 6371;
  const dLat = toRadians(lat2 - lat1);
  const dLon = toRadians(lon2 - lon1);
  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos(toRadians(lat1)) *
      Math.cos(toRadians(lat2)) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

async function geocodeAreaToPoint(areaName) {
  const name = String(areaName || "").trim();
  if (!name) return { ok: false, reason: "empty-area-name" };

  const geo = new URL(config.weatherGeocodeBaseUrl);
  geo.searchParams.set("name", name);
  geo.searchParams.set("count", "1");
  geo.searchParams.set("language", "en");
  geo.searchParams.set("format", "json");

  const geoResp = await httpGetJson(geo, config.analyticsTimeoutMs);
  if (!geoResp.ok) {
    return { ok: false, reason: `geocoding-failed:${geoResp.status}` };
  }

  const point = geoResp.data?.results?.[0];
  if (!point) {
    return { ok: false, reason: "no-geocoding-result" };
  }

  const lat = Number(point.latitude);
  const lon = Number(point.longitude);
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
    return { ok: false, reason: "invalid-geocode-coordinates" };
  }

  return {
    ok: true,
    name: String(point.name || name),
    lat,
    lon,
  };
}

async function countComplaintsWithinRadiusKm({
  complaintsCol,
  complaintMatch,
  centerLat,
  centerLon,
  radiusKm,
}) {
  const rows = await complaintsCol
    .find(complaintMatch, { projection: { location: 1 } })
    .toArray();

  let withCoordinates = 0;
  let withinRadius = 0;
  for (const row of rows) {
    const point = parseLocationCoordinates(row.location);
    if (!point) continue;
    withCoordinates += 1;
    const distKm = haversineKm(centerLat, centerLon, point.lat, point.lon);
    if (distKm <= radiusKm) withinRadius += 1;
  }

  return {
    count: withinRadius,
    scanned: rows.length,
    withCoordinates,
    radiusKm,
  };
}

function isLikelyComplaintCountQuestion(question) {
  const q = String(question || "");
  return (
    /\b(how many|count|total|number of)\b/i.test(q) &&
    /\bcomplaints?\b/i.test(q)
  );
}

async function answerGovEmployeeFromDb(question) {
  const db = await getDb();
  const complaintsCol = db.collection(config.complaintCollectionName);
  const departmentsCol = db.collection(config.departmentCollectionName);

  const complaintCount = await complaintsCol.countDocuments({});
  if (complaintCount === 0) {
    return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
  }

  const intent = parseGovEmpIntent(question);
  const statusRegex = statusRegexForIntent(intent.status);
  const complaintMatch = {};
  if (statusRegex) complaintMatch.status = { $regex: statusRegex };

  if (intent.wantsDeptWise) {
    const rows = await complaintsCol
      .aggregate([
        { $match: complaintMatch },
        {
          $lookup: {
            from: config.departmentCollectionName,
            localField: "departmentId",
            foreignField: "_id",
            as: "department",
          },
        },
        {
          $addFields: {
            departmentName: {
              $ifNull: [{ $arrayElemAt: ["$department.name", 0] }, "Unassigned"],
            },
          },
        },
        { $group: { _id: "$departmentName", count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: 10 },
      ])
      .toArray();

    if (!rows.length) return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
    const lines = rows.map((r) => `${r._id}: ${r.count}`).join(", ");
    const factual = `Complaint counts by department: ${lines}.`;
    const answer = await generateGovEmpAnswer(question, factual);
    return { answer, citations: [], mode: "govemp_db" };
  }

  if (intent.wantsAreaWise) {
    const rows = await complaintsCol
      .aggregate([
        { $match: complaintMatch },
        {
          $lookup: {
            from: config.areaCollectionName,
            localField: "areaId",
            foreignField: "_id",
            as: "area",
          },
        },
        {
          $addFields: {
            areaName: { $ifNull: [{ $arrayElemAt: ["$area.name", 0] }, "$location"] },
          },
        },
        { $group: { _id: "$areaName", count: { $sum: 1 } } },
        { $sort: { count: -1 } },
        { $limit: 10 },
      ])
      .toArray();

    if (!rows.length) return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
    const lines = rows.map((r) => `${r._id || "Unknown"}: ${r.count}`).join(", ");
    const factual = `Complaint counts by area: ${lines}.`;
    const answer = await generateGovEmpAnswer(question, factual);
    return { answer, citations: [], mode: "govemp_db" };
  }

  let departmentDoc = null;
  let areaDoc = null;
  let areaGeoPoint = null;
  const hasDepartmentHint = intent.departmentName || /\bdepartment\b|\bdept\b/i.test(question);
  const extractedAreaPhrase = intent.areaName || extractAreaLikePhrase(question);
  const hasAreaHint =
    !!extractedAreaPhrase ||
    /\barea\b|\bward\b|\bzone\b|\bpanchayat\b|\bcity\b|\bvillage\b|\blocality\b/i.test(question);
  const genericCountIntent = isLikelyComplaintCountQuestion(question);

  let deptCandidates = [];
  if (hasDepartmentHint || genericCountIntent) {
    deptCandidates = await departmentsCol.find({}, { projection: { name: 1 } }).toArray();
  }

  if (intent.departmentName && deptCandidates?.length) {
    departmentDoc =
      deptCandidates.find((d) => textContains(d.name, intent.departmentName)) || null;
  }

  if (!departmentDoc && deptCandidates?.length && hasDepartmentHint) {
    departmentDoc = findBestNamedEntityInQuestion(question, deptCandidates);
  }

  if (!departmentDoc && genericCountIntent && deptCandidates?.length) {
    const deptMatch = findBestNamedEntityMatch(question, deptCandidates);
    if (deptMatch.score > 0) {
      departmentDoc = deptMatch.entity;
    }
  }

  if (hasDepartmentHint && !departmentDoc) {
    return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
  }
  if (departmentDoc) {
    complaintMatch.departmentId = departmentDoc._id;
  }

  if (extractedAreaPhrase) {
    const geo = await geocodeAreaToPoint(extractedAreaPhrase);
    if (!geo.ok) {
      return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
    }
    areaDoc = { name: extractedAreaPhrase };
    areaGeoPoint = geo;
  } else if (hasAreaHint) {
    return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };
  }

  if (intent.contractorName) {
    complaintMatch.$or = [
      { contractorName: { $regex: intent.contractorName, $options: "i" } },
      { contractor: { $regex: intent.contractorName, $options: "i" } },
      { assignedContractor: { $regex: intent.contractorName, $options: "i" } },
    ];
  }

  let count = 0;
  let areaRadiusStats = null;
  if (areaGeoPoint) {
    areaRadiusStats = await countComplaintsWithinRadiusKm({
      complaintsCol,
      complaintMatch,
      centerLat: areaGeoPoint.lat,
      centerLon: areaGeoPoint.lon,
      radiusKm: DEFAULT_AREA_RADIUS_KM,
    });
    count = areaRadiusStats.count;
  } else {
    count = await complaintsCol.countDocuments(complaintMatch);
  }

  if (!count) return { answer: NO_DATA_TEXT, citations: [], mode: "govemp_db" };

  const filters = [];
  if (intent.status !== "any") filters.push(intent.status);
  if (departmentDoc?.name) filters.push(`department ${departmentDoc.name}`);
  if (areaDoc?.name) filters.push(`area ${areaDoc.name}`);
  else if (intent.areaName) filters.push(`area ${intent.areaName}`);
  if (intent.contractorName) filters.push(`contractor ${intent.contractorName}`);

  const suffix = filters.length ? ` for ${filters.join(", ")}` : "";
  const radiusNote = areaRadiusStats
    ? ` within ${DEFAULT_AREA_RADIUS_KM}km radius of ${areaGeoPoint.name || areaDoc?.name || "requested location"}`
    : "";
  const factual = `Total complaints${suffix}${radiusNote}: ${count}.`;
  const answer = await generateGovEmpAnswer(question, factual);
  return { answer, citations: [], mode: "govemp_db" };
}

async function generateGovEmpAnswer(question, factualLine) {
  try {
    const prompt = [
      "You are a government complaint assistant.",
      "Respond in 1-2 clear sentences for an employee dashboard.",
      "Use only the factual data provided below.",
      "Do not invent numbers or entities.",
      "",
      `Question: ${question}`,
      `Factual data: ${factualLine}`,
    ].join("\n");

    const text = await generateTextFromPrompt(prompt, {
      temperature: 0.1,
      maxTokens: Math.min(config.ollamaNumPredict, 96),
    });
    return text || factualLine;
  } catch (err) {
    console.error(
      "[RAG] GovEmp generation failed, using factual fallback:",
      err.message
    );
    return factualLine;
  }
}

function getEffectiveProvider() {
  if (config.prodEnv) return "groq";
  const configured = String(config.llmProvider || "").toLowerCase();
  return configured === "groq" ? "groq" : "ollama";
}

function capTokensForProvider(requested, defaultLimit = 220) {
  const provider = getEffectiveProvider();
  const fallback = Number(defaultLimit || 220);
  const asked = Number(requested || fallback);
  if (provider === "groq") {
    return Math.min(asked, Number(config.groqMaxTokens || 256));
  }
  return Math.min(asked, Number(config.ollamaNumPredict || 256));
}

async function generateTextFromPrompt(prompt, { temperature, maxTokens }) {
  const provider = getEffectiveProvider();

  if (provider === "groq") {
    if (!config.groqApiKey) {
      throw new Error("GROQ_API_KEY is required when prod_env=true");
    }

    const completion = await callGroqChatCompletions({
      baseUrl: config.groqApiBaseUrl,
      timeoutMs: config.groqTimeoutMs,
      apiKey: config.groqApiKey,
      payload: {
        model: config.groqChatModel,
        messages: [{ role: "user", content: prompt }],
        temperature,
        max_tokens: Math.min(maxTokens, config.groqMaxTokens),
      },
    });

    const text = String(completion?.choices?.[0]?.message?.content || "").trim();
    if (!text) throw new Error("Groq returned an empty response");
    return text;
  }

  const completion = await callOllamaGenerate({
    baseUrl: config.ollamaBaseUrl,
    timeoutMs: config.ollamaTimeoutMs,
    payload: {
      model: config.ollamaChatModel,
      prompt,
      stream: false,
      options: {
        temperature,
        num_predict: maxTokens,
      },
    },
  });

  return String(completion.response || "").trim();
}

function callOllamaGenerate({ baseUrl, payload, timeoutMs }) {
  return new Promise((resolve, reject) => {
    const endpoint = new URL("/api/generate", baseUrl);
    const client = endpoint.protocol === "https:" ? https : http;
    const body = JSON.stringify(payload);

    const req = client.request(
      endpoint,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
        },
      },
      (res) => {
        let raw = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          raw += chunk;
        });
        res.on("end", () => {
          if ((res.statusCode || 500) >= 400) {
            reject(
              new Error(`Ollama generation failed: ${res.statusCode} ${raw}`)
            );
            return;
          }
          try {
            resolve(JSON.parse(raw));
          } catch (err) {
            reject(new Error(`Invalid Ollama JSON response: ${err.message}`));
          }
        });
      }
    );

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Ollama request timed out after ${timeoutMs}ms`));
    });
    req.on("error", (err) => reject(err));
    req.write(body);
    req.end();
  });
}

function callGroqChatCompletions({ baseUrl, payload, timeoutMs, apiKey }) {
  return new Promise((resolve, reject) => {
    const endpoint = new URL(baseUrl.replace(/\/+$/, "") + "/chat/completions");
    const client = endpoint.protocol === "https:" ? https : http;
    const body = JSON.stringify(payload);

    const req = client.request(
      endpoint,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
          "Content-Length": Buffer.byteLength(body),
        },
      },
      (res) => {
        let raw = "";
        res.setEncoding("utf8");
        res.on("data", (chunk) => {
          raw += chunk;
        });
        res.on("end", () => {
          if ((res.statusCode || 500) >= 400) {
            reject(new Error(`Groq generation failed: ${res.statusCode} ${raw}`));
            return;
          }
          try {
            resolve(JSON.parse(raw));
          } catch (err) {
            reject(new Error(`Invalid Groq JSON response: ${err.message}`));
          }
        });
      }
    );

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Groq request timed out after ${timeoutMs}ms`));
    });
    req.on("error", (err) => reject(err));
    req.write(body);
    req.end();
  });
}

function normalizeLanguage(language) {
  const code = String(language || "en").trim().toLowerCase();
  const supported = {
    en: "English",
    hi: "Hindi",
    kn: "Kannada",
    ta: "Tamil",
    te: "Telugu",
    ml: "Malayalam",
  };
  if (supported[code]) {
    return { code, label: supported[code], fallbackApplied: false };
  }
  return { code: "en", label: "English", fallbackApplied: true };
}

function clamp(n, min, max) {
  return Math.max(min, Math.min(max, n));
}

function toPct(numerator, denominator) {
  if (!denominator) return 0;
  return (numerator / denominator) * 100;
}

function round2(value) {
  return Math.round(Number(value || 0) * 100) / 100;
}

function isPendingStatus(status) {
  return /registered|open|in[-_\s]?progress|pending|assigned|investigating|reopened/i.test(
    String(status || "")
  );
}

function isResolvedStatus(status) {
  return /resolved|closed|completed|done|solved|fixed/i.test(
    String(status || "")
  );
}

function detectCauseFromText(doc) {
  const combined = [doc.title, doc.complaint, doc.summarized_complaint]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();

  const rules = [
    { key: "Potholes and Road Damage", re: /pothole|road damage|road crack|tar|asphalt/ },
    { key: "Water Supply and Leakage", re: /water leak|leakage|low pressure|no water|pipeline/ },
    { key: "Drainage and Waterlogging", re: /drain|waterlog|flood|sewer overflow|storm water/ },
    { key: "Streetlight and Electricity", re: /streetlight|power cut|electric|transformer|outage/ },
    { key: "Garbage and Sanitation", re: /garbage|waste|sanitation|trash|dump/ },
    { key: "Public Health and Mosquito", re: /mosquito|health|stagnant water|dengue/ },
  ];

  for (const rule of rules) {
    if (rule.re.test(combined)) return rule.key;
  }
  return "General Civic Issues";
}

function getCauseList(doc) {
  const categories = Array.isArray(doc.issue_category) ? doc.issue_category : [];
  const cleaned = categories.map((v) => String(v || "").trim()).filter(Boolean);
  if (cleaned.length) return cleaned;
  return [detectCauseFromText(doc)];
}

function computeWeatherSignals(weather) {
  if (!weather || !weather.available) {
    return {
      available: false,
      summary: "Weather signal unavailable",
      riskHints: [],
      triggerLevel: "low",
    };
  }

  const desc = String(weather.description || "").toLowerCase();
  const main = String(weather.main || "").toLowerCase();
  const temp = Number(weather.tempC || 0);
  const rain1h = Number(weather.rain1hMm || 0);
  const month = new Date().getUTCMonth() + 1;

  const hints = [];
  let triggerLevel = "low";

  const monsoonMonth = month >= 6 && month <= 10;
  const rainy = rain1h >= 2 || /rain|storm|thunder/.test(desc) || main === "rain";
  if (rainy || monsoonMonth) {
    hints.push("Higher risk of potholes, waterlogging, and drainage choke points");
    hints.push("Pre-position road patch teams and storm-drain desilting crews");
    triggerLevel = rainy ? "high" : "medium";
  }

  if (temp >= 36) {
    hints.push("Higher risk of transformer stress and peak water demand");
    hints.push("Schedule preventive checks for feeders and water pumping stations");
    triggerLevel = triggerLevel === "high" ? "high" : "medium";
  }

  if (/wind|squall/.test(desc)) {
    hints.push("Potential treefall/line-fault incidents; trim vulnerable corridors");
    triggerLevel = "high";
  }

  if (!hints.length) {
    hints.push("No major weather-driven civic spike expected in the immediate horizon");
  }

  return {
    available: true,
    summary: `${weather.main || "Weather"} (${weather.description || ""}), ${round2(
      weather.tempC
    )}C`,
    triggerLevel,
    riskHints: hints,
  };
}

function dayKey(dateValue) {
  const d = new Date(dateValue);
  if (Number.isNaN(d.getTime())) return null;
  return d.toISOString().slice(0, 10);
}

function buildForecastFromDailyCounts(dailyCounts, windowDays) {
  const weights = [0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9];
  const series = dailyCounts.slice(-7);
  if (!series.length) {
    return { predictedNext7dCount: 0, trendPercent: 0 };
  }

  let weightedSum = 0;
  let weightTotal = 0;
  for (let i = 0; i < series.length; i += 1) {
    const w = weights[weights.length - series.length + i];
    weightedSum += series[i] * w;
    weightTotal += w;
  }
  const baseDaily = weightTotal ? weightedSum / weightTotal : 0;

  const head = series.slice(0, Math.max(1, Math.floor(series.length / 2)));
  const tail = series.slice(Math.max(1, Math.floor(series.length / 2)));
  const headAvg = head.reduce((a, b) => a + b, 0) / head.length;
  const tailAvg = tail.reduce((a, b) => a + b, 0) / tail.length;
  const trendPercent = headAvg > 0 ? ((tailAvg - headAvg) / headAvg) * 100 : 0;

  const trendFactor = clamp(1 + trendPercent / 100, 0.6, 1.8);
  const horizon = Math.max(3, Math.min(10, Number(windowDays || 7)));
  const predictedNext7dCount = Math.max(0, Math.round(baseDaily * 7 * trendFactor));

  return {
    predictedNext7dCount,
    trendPercent: round2(trendPercent),
    baselineDaily: round2(baseDaily),
    horizon,
  };
}

function confidenceScore(sampleSize, trendPercent, minSampleSize) {
  const sampleComponent = clamp(sampleSize / Math.max(minSampleSize, 1), 0.2, 1);
  const stabilityComponent = clamp(1 - Math.abs(trendPercent) / 200, 0.35, 1);
  return round2(sampleComponent * 0.7 + stabilityComponent * 0.3);
}

function riskLevelFromSignals({ pendingPercent, trendPercent, weatherTrigger }) {
  let score = 0;
  if (pendingPercent >= 60) score += 2;
  else if (pendingPercent >= 40) score += 1;
  if (trendPercent >= 20) score += 2;
  else if (trendPercent >= 8) score += 1;
  if (weatherTrigger === "high") score += 2;
  else if (weatherTrigger === "medium") score += 1;

  if (score >= 5) return "high";
  if (score >= 3) return "medium";
  return "low";
}

function topEntries(counterMap, limit) {
  return Array.from(counterMap.entries())
    .sort((a, b) => b[1] - a[1])
    .slice(0, limit)
    .map(([name, count]) => ({ name, count }));
}

function buildPreventiveActions({ topCause, riskLevel, weatherSignals, scopeLabel }) {
  const actions = [];
  const causeLower = String(topCause || "").toLowerCase();

  if (causeLower.includes("road") || causeLower.includes("pothole")) {
    actions.push(
      "Run rapid road-condition audits in complaint hotspots and schedule preventive patching before peak rainfall."
    );
  }
  if (causeLower.includes("water") || causeLower.includes("drain")) {
    actions.push(
      "Start weekly drain desilting and pipeline leak checks in high-incidence wards to reduce repeat complaints."
    );
  }
  if (causeLower.includes("electric") || causeLower.includes("streetlight")) {
    actions.push(
      "Perform feeder and streetlight preventive maintenance in top complaint corridors and stock critical spare parts."
    );
  }
  if (!actions.length) {
    actions.push(
      "Deploy targeted preventive inspections in the top 3 complaint areas and close recurring root causes within 72 hours."
    );
  }

  if (weatherSignals?.triggerLevel === "high") {
    actions.push(
      "Activate monsoon/storm readiness: emergency crews, dewatering pumps, and public alert messaging for vulnerable areas."
    );
  }
  if (riskLevel === "high") {
    actions.push(
      `For ${scopeLabel}, trigger a 7-day escalation plan with daily closure targets and field verification.`
    );
  }

  return actions.slice(0, 3);
}

function tryParseHeadlineDescription(text) {
  const raw = String(text || "").trim();
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    if (parsed && parsed.headline && parsed.description) {
      return {
        headline: String(parsed.headline).trim(),
        description: String(parsed.description).trim(),
      };
    }
  } catch (_err) {
    // no-op; fall through to plain text parsing
  }

  const parts = raw
    .split(/\n+/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (!parts.length) return null;
  if (parts.length === 1) {
    return {
      headline: "Predictive civic risk outlook",
      description: parts[0],
    };
  }
  return {
    headline: parts[0].replace(/^headline\s*[:\-]?/i, "").trim(),
    description: parts.slice(1).join(" ").replace(/^description\s*[:\-]?/i, "").trim(),
  };
}

function normalizeQuestionCandidate(value) {
  const text = String(value || "")
    .replace(/^[\-\d\.)\s]+/, "")
    .replace(/^['\"`]+|['\"`]+$/g, "")
    .trim();
  if (!text) return "";
  const withMark = /[?]$/.test(text) ? text : `${text}?`;
  return withMark;
}

function tryParseQuestionArray(rawText) {
  const raw = String(rawText || "").trim();
  if (!raw) return [];

  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed
        .map((item) => normalizeQuestionCandidate(item))
        .filter(Boolean);
    }
  } catch (_err) {
    // no-op; continue with line-based parsing
  }

  return raw
    .split(/\n+/)
    .map((line) => normalizeQuestionCandidate(line))
    .filter(Boolean);
}

function pickTwoDistinctQuestions(candidates, baseQuestion) {
  const baseNorm = normalizeMatchText(baseQuestion);
  const uniq = [];
  const seen = new Set();

  for (const c of candidates) {
    const question = normalizeQuestionCandidate(c);
    const key = normalizeMatchText(question);
    if (!question || !key || key === baseNorm || seen.has(key)) continue;
    seen.add(key);
    uniq.push(question);
  }

  for (let i = uniq.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = uniq[i];
    uniq[i] = uniq[j];
    uniq[j] = tmp;
  }

  return uniq.slice(0, 2);
}

function fallbackRecommendedQuestions(question) {
  const rawTopic =
    String(question || "")
      .replace(/[?]/g, "")
      .trim() || "this issue";
  const topic = rawTopic.length > 80 ? `${rawTopic.slice(0, 77)}...` : rawTopic;

  const perimeterTemplates = [
    `Which nearby areas are showing similar patterns to ${topic}?`,
    `How does ${topic} compare across departments or wards this month?`,
    `What related complaint categories are clustered around ${topic}?`,
    `Where is the geographic perimeter of ${topic} expanding most quickly?`,
  ];
  const depthTemplates = [
    `What is the root-cause depth behind ${topic}, and which factor appears first?`,
    `Which service bottleneck is most responsible for delays related to ${topic}?`,
    `What preventive action would reduce repeat complaints linked to ${topic}?`,
    `How has closure quality changed over time for issues like ${topic}?`,
  ];

  const p = perimeterTemplates[Math.floor(Math.random() * perimeterTemplates.length)];
  const d = depthTemplates[Math.floor(Math.random() * depthTemplates.length)];
  return pickTwoDistinctQuestions([p, d], question);
}

function randomPickDistinct(values, count) {
  const arr = Array.isArray(values) ? [...values] : [];
  for (let i = arr.length - 1; i > 0; i -= 1) {
    const j = Math.floor(Math.random() * (i + 1));
    const tmp = arr[i];
    arr[i] = arr[j];
    arr[j] = tmp;
  }
  return arr.slice(0, Math.max(0, Number(count || 0)));
}

function buildGovEmpRecommendedQuestions(question) {
  const intent = parseGovEmpIntent(question);
  const areaText = intent.areaName || extractAreaLikePhrase(question) || "this area";
  const deptText = intent.departmentName || "this department";
  const contractorText = intent.contractorName || "this contractor";

  const genericPool = [
    "How many pending complaints are there now?",
    "How many resolved complaints are there now?",
    "How many complaints were registered today?",
    "Which department has the highest pending complaints?",
    "How many complaints are open by area?",
    "How many complaints are open by department?",
  ];

  const areaPool = [
    `How many complaints are there in ${areaText}?`,
    `How many resolved complaints are there in ${areaText}?`,
    `How many pending complaints are there in ${areaText}?`,
    `Which complaint type is highest in ${areaText}?`,
  ];

  const departmentPool = [
    `How many complaints are there in ${deptText}?`,
    `How many pending complaints are there in ${deptText}?`,
    `How many resolved complaints are there in ${deptText}?`,
    `What is the closure rate in ${deptText}?`,
  ];

  const contractorPool = [
    `How many complaints are assigned to ${contractorText}?`,
    `How many pending complaints are assigned to ${contractorText}?`,
    `How many resolved complaints are completed by ${contractorText}?`,
  ];

  const statusPool =
    intent.status === "pending"
      ? [
          "How many pending complaints are open by area?",
          "Which department has the most pending complaints?",
          "How many pending complaints are older than 7 days?",
        ]
      : intent.status === "resolved"
      ? [
          "How many resolved complaints were closed this week?",
          "Which department closed the most complaints this week?",
          "How many resolved complaints are there by area?",
        ]
      : [];

  const combined = [
    ...genericPool,
    ...(intent.areaName || areaText !== "this area" ? areaPool : []),
    ...(intent.departmentName ? departmentPool : []),
    ...(intent.contractorName ? contractorPool : []),
    ...statusPool,
  ];

  const normalized = pickTwoDistinctQuestions(
    randomPickDistinct(combined, Math.max(8, combined.length)),
    question
  );

  if (normalized.length >= 2) return normalized;
  return randomPickDistinct(genericPool, 2);
}

async function generateRecommendedQuestions(question, answer, mode) {
  if (String(mode || "").toLowerCase() === "govemp") {
    return buildGovEmpRecommendedQuestions(question);
  }

  const prompt = [
    "You create follow-up questions for a civic complaint assistant.",
    "Generate 6 diverse follow-up questions related to the user question.",
    "Include a blend of: (1) perimeter/scope expansion and (2) service-depth/root-cause analysis.",
    "Keep each question under 20 words.",
    "Questions must be practical for governance workflows and must not repeat the original question.",
    "Return ONLY a JSON array of strings.",
    "",
    `Mode: ${mode || "rag"}`,
    `Original question: ${question}`,
    `Current answer summary: ${String(answer || "").slice(0, 320)}`,
    `Variation seed: ${Date.now()}-${Math.floor(Math.random() * 1000000)}`,
  ].join("\n");

  try {
    const raw = await generateTextFromPrompt(prompt, {
      temperature: 0.7,
      maxTokens: capTokensForProvider(220),
    });

    const parsed = tryParseQuestionArray(raw);
    const picked = pickTwoDistinctQuestions(parsed, question);
    if (picked.length >= 2) return picked;

    const fallback = fallbackRecommendedQuestions(question);
    if (picked.length) {
      return pickTwoDistinctQuestions([...picked, ...fallback], question).slice(0, 2);
    }
    return fallback;
  } catch (err) {
    console.error("[RAG] Failed generating recommended questions:", err.message);
    return fallbackRecommendedQuestions(question);
  }
}

function weatherCodeLabel(code) {
  const n = Number(code);
  if (n === 0) return { main: "Clear", description: "clear sky" };
  if ([1, 2, 3].includes(n)) return { main: "Clouds", description: "partly cloudy" };
  if ([45, 48].includes(n)) return { main: "Fog", description: "fog" };
  if ([51, 53, 55, 56, 57].includes(n)) return { main: "Drizzle", description: "drizzle" };
  if ([61, 63, 65, 66, 67, 80, 81, 82].includes(n)) {
    return { main: "Rain", description: "rain" };
  }
  if ([71, 73, 75, 77, 85, 86].includes(n)) return { main: "Snow", description: "snow" };
  if ([95, 96, 99].includes(n)) return { main: "Storm", description: "thunderstorm" };
  return { main: "Weather", description: "mixed" };
}

function httpGetJson(endpointUrl, timeoutMs) {
  return new Promise((resolve) => {
    const client = endpointUrl.protocol === "https:" ? https : http;
    const req = client.request(endpointUrl, { method: "GET" }, (res) => {
      let raw = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        raw += chunk;
      });
      res.on("end", () => {
        if ((res.statusCode || 500) >= 400) {
          resolve({ ok: false, status: res.statusCode || 500, error: raw });
          return;
        }
        try {
          resolve({ ok: true, data: JSON.parse(raw) });
        } catch (err) {
          resolve({ ok: false, status: 500, error: `JSON parse failed: ${err.message}` });
        }
      });
    });

    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error("Weather request timed out"));
    });
    req.on("error", (err) => {
      resolve({ ok: false, status: 500, error: err.message });
    });
    req.end();
  });
}

async function fetchOpenMeteoSignal() {
  let latitude = Number(config.weatherLatitude);
  let longitude = Number(config.weatherLongitude);

  if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
    const geo = new URL(config.weatherGeocodeBaseUrl);
    geo.searchParams.set("name", config.weatherLocationQuery || "Bengaluru");
    geo.searchParams.set("count", "1");
    geo.searchParams.set("language", "en");
    geo.searchParams.set("format", "json");

    const geoResp = await httpGetJson(geo, config.analyticsTimeoutMs);
    if (!geoResp.ok) {
      return {
        available: false,
        reason: `Open-Meteo geocoding failed: ${geoResp.status}`,
      };
    }

    const point = geoResp.data?.results?.[0];
    if (!point) {
      return { available: false, reason: "Open-Meteo geocoding returned no results" };
    }
    latitude = Number(point.latitude);
    longitude = Number(point.longitude);
  }

  const forecastUrl = new URL(config.weatherApiBaseUrl);
  forecastUrl.searchParams.set("latitude", String(latitude));
  forecastUrl.searchParams.set("longitude", String(longitude));
  forecastUrl.searchParams.set(
    "current",
    "temperature_2m,precipitation,rain,weather_code,wind_speed_10m"
  );
  forecastUrl.searchParams.set("timezone", config.weatherTimezone || "auto");

  const forecastResp = await httpGetJson(forecastUrl, config.analyticsTimeoutMs);
  if (!forecastResp.ok) {
    return {
      available: false,
      reason: `Open-Meteo forecast failed: ${forecastResp.status}`,
    };
  }

  const current = forecastResp.data?.current;
  if (!current) {
    return { available: false, reason: "Open-Meteo forecast missing current weather" };
  }

  const mapped = weatherCodeLabel(current.weather_code);
  return {
    available: true,
    main: mapped.main,
    description: mapped.description,
    tempC: Number(current.temperature_2m || 0),
    rain1hMm: Number(current.rain || current.precipitation || 0),
  };
}

async function fetchWeatherSignal() {
  const provider = String(config.weatherApiProvider || "open-meteo").toLowerCase();
  if (provider === "open-meteo") {
    return fetchOpenMeteoSignal();
  }
  return { available: false, reason: `Unsupported weather provider: ${provider}` };
}

async function generateAnalyticsNarrative({
  languageLabel,
  analyticsSummary,
  evidence,
  recommendations,
}) {
  const prompt = [
    "You are a municipal predictive-analytics communication assistant.",
    `Respond in ${languageLabel}.`,
    "You must use only the provided numeric evidence. Do not invent any number, cause, or entity.",
    "Return strict JSON with fields: headline, description.",
    "headline must be short and action oriented.",
    "description must mention: time window, predicted complaint load, top cause, and 1-2 prevention actions.",
    "",
    `Analytics summary: ${analyticsSummary}`,
    `Evidence JSON: ${JSON.stringify(evidence)}`,
    `Recommendations: ${recommendations.join(" | ")}`,
  ].join("\n");

  const raw = await generateTextFromPrompt(prompt, {
    temperature: 0.1,
    maxTokens: Math.min(config.groqMaxTokens || 256, 240),
  });

  return tryParseHeadlineDescription(raw);
}

async function getPredictiveAnalytics({ departmentName, language, windowDays }) {
  const totalStart = Date.now();
  const normalizedLang = normalizeLanguage(language);
  const resolvedWindowDays = Math.max(
    3,
    Math.min(30, Number(windowDays || config.analyticsDefaultWindowDays || 7))
  );

  const db = await getDb();
  const complaintsCol = db.collection(config.complaintCollectionName);
  const departmentsCol = db.collection(config.departmentCollectionName);

  let resolvedDepartment = null;
  let mode = "commissioner";
  if (departmentName) {
    const deptCandidates = await departmentsCol
      .find({}, { projection: { name: 1 } })
      .toArray();
    resolvedDepartment =
      deptCandidates.find((d) => textContains(d.name, departmentName)) || null;
    if (!resolvedDepartment) {
      return {
        headline: "Department not found for analytics",
        description:
          "No matching department was found. Check the department name and try again.",
        evidence: {
          windowDays: resolvedWindowDays,
          sampleSize: 0,
          confidence: 0,
          weatherSignals: { available: false, summary: "Not evaluated" },
          forecast: { predictedNext7dCount: 0, probableTopCause: "N/A", riskLevel: "low" },
        },
        scope: {
          mode,
          departmentResolved: null,
          filtersApplied: ["departmentName"],
          language: normalizedLang.code,
          fallbackLanguageApplied: normalizedLang.fallbackApplied,
        },
        generatedAt: new Date().toISOString(),
      };
    }
    mode = "department";
  }

  const since = new Date(Date.now() - resolvedWindowDays * 24 * 60 * 60 * 1000);
  const match = {
    $or: [{ date: { $gte: since } }, { lastupdate: { $gte: since } }],
  };
  if (resolvedDepartment?._id) {
    match.departmentId = resolvedDepartment._id;
  }

  const dbStart = Date.now();
  const rows = await complaintsCol
    .aggregate([
      { $match: match },
      {
        $lookup: {
          from: config.areaCollectionName,
          localField: "areaId",
          foreignField: "_id",
          as: "area",
        },
      },
      {
        $project: {
          title: 1,
          complaint: 1,
          summarized_complaint: 1,
          issue_category: 1,
          status: 1,
          date: 1,
          lastupdate: 1,
          priority_factor: 1,
          location: 1,
          areaName: { $ifNull: [{ $arrayElemAt: ["$area.name", 0] }, "$location"] },
        },
      },
    ])
    .toArray();
  const dbMs = Date.now() - dbStart;

  if (!rows.length) {
    return {
      headline: "Insufficient recent complaints for prediction",
      description: `Using the last ${resolvedWindowDays} days, there is not enough complaint data to generate a reliable forecast.`,
      evidence: {
        windowDays: resolvedWindowDays,
        sampleSize: 0,
        totals: { complaints: 0, pending: 0, resolved: 0 },
        pendingPercent: 0,
        resolvedPercent: 0,
        trendPercent: 0,
        topCauses: [],
        topAreas: [],
        weatherSignals: { available: false, summary: "Not evaluated" },
        confidence: 0,
        forecast: { predictedNext7dCount: 0, probableTopCause: "N/A", riskLevel: "low" },
      },
      scope: {
        mode,
        departmentResolved: resolvedDepartment?.name || null,
        filtersApplied: resolvedDepartment ? ["department"] : ["all_departments"],
        language: normalizedLang.code,
        fallbackLanguageApplied: normalizedLang.fallbackApplied,
      },
      generatedAt: new Date().toISOString(),
      timings: { dbMs, weatherMs: 0, llmMs: 0, totalMs: Date.now() - totalStart },
    };
  }

  const causeCounter = new Map();
  const areaCounter = new Map();
  let pendingCount = 0;
  let resolvedCount = 0;
  let prioritySum = 0;
  const dailyMap = new Map();

  for (let i = 0; i < resolvedWindowDays; i += 1) {
    const d = new Date(since.getTime() + i * 24 * 60 * 60 * 1000);
    dailyMap.set(d.toISOString().slice(0, 10), 0);
  }

  for (const row of rows) {
    if (isPendingStatus(row.status)) pendingCount += 1;
    if (isResolvedStatus(row.status)) resolvedCount += 1;
    prioritySum += Number(row.priority_factor || 0);

    const areaName = String(row.areaName || "Unknown Area").trim() || "Unknown Area";
    areaCounter.set(areaName, (areaCounter.get(areaName) || 0) + 1);

    const causes = getCauseList(row);
    for (const cause of causes) {
      causeCounter.set(cause, (causeCounter.get(cause) || 0) + 1);
    }

    const createdKey = dayKey(row.date || row.lastupdate);
    if (createdKey && dailyMap.has(createdKey)) {
      dailyMap.set(createdKey, (dailyMap.get(createdKey) || 0) + 1);
    }
  }

  const sampleSize = rows.length;
  const pendingPercent = round2(toPct(pendingCount, sampleSize));
  const resolvedPercent = round2(toPct(resolvedCount, sampleSize));
  const topCauses = topEntries(causeCounter, 5);
  const topAreas = topEntries(areaCounter, 5);
  const dailySeries = Array.from(dailyMap.values());
  const forecast = buildForecastFromDailyCounts(dailySeries, resolvedWindowDays);

  const weatherStart = Date.now();
  const weatherRaw = await fetchWeatherSignal();
  const weatherMs = Date.now() - weatherStart;
  const weatherSignals = computeWeatherSignals(weatherRaw);

  const riskLevel = riskLevelFromSignals({
    pendingPercent,
    trendPercent: forecast.trendPercent,
    weatherTrigger: weatherSignals.triggerLevel,
  });

  const probableTopCause = topCauses.length ? topCauses[0].name : "General Civic Issues";
  const conf = Math.max(
    Number(config.analyticsConfidenceFloor || 0.45),
    confidenceScore(sampleSize, forecast.trendPercent, config.analyticsMinSampleSize)
  );

  const scopeLabel =
    mode === "department"
      ? `${resolvedDepartment.name} department`
      : "all municipal departments";
  const actions = buildPreventiveActions({
    topCause: probableTopCause,
    riskLevel,
    weatherSignals,
    scopeLabel,
  });

  const evidence = {
    windowDays: resolvedWindowDays,
    sampleSize,
    totals: {
      complaints: sampleSize,
      pending: pendingCount,
      resolved: resolvedCount,
    },
    pendingPercent,
    resolvedPercent,
    trendPercent: forecast.trendPercent,
    avgPriorityFactor: round2(prioritySum / Math.max(sampleSize, 1)),
    topCauses,
    topAreas,
    weatherSignals,
    confidence: round2(conf),
    forecast: {
      predictedNext7dCount: forecast.predictedNext7dCount,
      probableTopCause,
      riskLevel,
    },
    recommendations: actions,
  };

  const analyticsSummary = [
    `Scope: ${scopeLabel}`,
    `Window: last ${resolvedWindowDays} days`,
    `Total complaints: ${sampleSize}`,
    `Pending: ${pendingPercent}%`,
    `Trend: ${forecast.trendPercent}%`,
    `Predicted next 7d: ${forecast.predictedNext7dCount}`,
    `Top cause: ${probableTopCause}`,
    `Risk level: ${riskLevel}`,
  ].join(" | ");

  let headline =
    mode === "department"
      ? `${resolvedDepartment.name}: proactive complaint-risk outlook`
      : "Citywide proactive complaint-risk outlook";
  let description = `Using the last ${resolvedWindowDays} days (${sampleSize} complaints), predicted incoming complaints for next 7 days are ${forecast.predictedNext7dCount}. The most likely cause is ${probableTopCause}. Suggested prevention: ${actions.join(" ")}`;

  const llmStart = Date.now();
  try {
    const narrative = await generateAnalyticsNarrative({
      languageLabel: normalizedLang.label,
      analyticsSummary,
      evidence,
      recommendations: actions,
    });
    if (narrative?.headline && narrative?.description) {
      headline = narrative.headline;
      description = narrative.description;
    }
  } catch (err) {
    console.error("[RAG] Predictive narrative fallback used:", err.message);
  }
  const llmMs = Date.now() - llmStart;

  return {
    headline,
    description,
    evidence,
    scope: {
      mode,
      departmentResolved: resolvedDepartment?.name || null,
      filtersApplied: resolvedDepartment ? ["department"] : ["all_departments"],
      language: normalizedLang.code,
      fallbackLanguageApplied: normalizedLang.fallbackApplied,
    },
    generatedAt: new Date().toISOString(),
    timings: {
      dbMs,
      weatherMs,
      llmMs,
      totalMs: Date.now() - totalStart,
    },
  };
}

module.exports = {
  answerWithRag,
  answerGovEmployeeFromDb,
  generateRecommendedQuestions,
  getPredictiveAnalytics,
};
