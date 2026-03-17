const express = require("express");
const {
  answerWithRag,
  answerGovEmployeeFromDb,
  getPredictiveAnalytics,
} = require("./ragService");

const router = express.Router();

router.post("/ragChat", async (req, res) => {
  try {
    const question = String(req.body?.question || "").trim();
    const topK = Number(req.body?.topK || 5);
    const govemp = Boolean(req.body?.govemp);
    console.log(
      `[RAG] Incoming /ragChat request | questionLen=${question.length} | topK=${topK} | govemp=${govemp}`
    );

    if (!question) {
      console.warn("[RAG] Rejected /ragChat request: empty question");
      return res.status(400).json({ error: "question is required" });
    }

    const result = govemp
      ? await answerGovEmployeeFromDb(question)
      : await answerWithRag(question, topK);
    console.log(
      `[RAG] Completed /ragChat | citations=${result.citations?.length || 0}`
    );
    return res.json(result);
  } catch (error) {
    console.error("ragChat error:", error);
    return res.status(500).json({
      error: "RAG request failed",
      detail: error.message || "Unknown error",
    });
  }
});

router.post("/predictiveAnalytics", async (req, res) => {
  try {
    const departmentName = String(req.body?.departmentName || "").trim();
    const language = String(req.body?.language || "").trim();
    const windowDaysRaw = Number(req.body?.windowDays || 0);
    const windowDays = Number.isFinite(windowDaysRaw)
      ? windowDaysRaw
      : undefined;

    if (!language) {
      return res.status(400).json({ error: "language is required" });
    }

    const result = await getPredictiveAnalytics({
      departmentName,
      language,
      windowDays,
    });
    return res.json(result);
  } catch (error) {
    console.error("predictiveAnalytics error:", error);
    return res.status(500).json({
      error: "Predictive analytics request failed",
      detail: error.message || "Unknown error",
    });
  }
});

module.exports = router;
