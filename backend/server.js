const express = require("express");
const cors = require("cors");
const config = require("./config");
const routes = require("./routes");
const { closeDb } = require("./db");

const app = express();

app.use(cors());
app.use(express.json({ limit: "1mb" }));

app.use((req, res, next) => {
  const start = Date.now();
  console.log(`[REQ] ${req.method} ${req.originalUrl}`);
  res.on("finish", () => {
    const latencyMs = Date.now() - start;
    console.log(
      `[RES] ${req.method} ${req.originalUrl} -> ${res.statusCode} (${latencyMs}ms)`
    );
  });
  next();
});

app.get("/health", (_req, res) => {
  res.json({ ok: true, service: "civic-governance-rag-api" });
});

app.use("/", routes);

const server = app.listen(config.port, () => {
  console.log(`RAG API listening on port ${config.port}`);
});

async function shutdown() {
  server.close(async () => {
    await closeDb();
    process.exit(0);
  });
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
