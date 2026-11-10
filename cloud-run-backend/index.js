// CS663 Week 11 Lab — Android Backend Integration to Tuned & Base Models
// Node.js + Express backend for Google Cloud Run.
//   POST /api/gemini            — base Gemini model
//   POST /api/fine-tuned-gemini — fine-tuned Gemini model from the Oct 30 exercise
// Alias: POST /api/base-gemini behaves exactly like /api/gemini (guide sample name).
// Both accept { "prompt": "...", "imageBase64": "...", "mimeType": "image/jpeg" }
// and return { "text": "..." }. Android client may send "text" instead of "prompt".
//
// Auth (lab guide 3.1):
//   Tuned: Vertex AI / Agent Platform via ADC from the Cloud Run service account.
//   Base: Vertex AI + ADC too (guide: this eliminates the AI Studio key).
//         If GEMINI_API_KEY is set it is used for base instead (Developer API).
// Why two Vertex locations: in project cs663-dog-training the base model
// gemini-3-flash-preview only answers via location "global" (verified Oct 30),
// while the tuned model only serves via its endpoint in location "us"
// (endpoint projects/917815040923/locations/us/endpoints/514008491845746688).
// So base and tuned each get their own client with its own location.
// Mock mode: MOCK_MODE=true returns a stub flagged mock:true. Never for submission.

import express from "express";
import { GoogleGenAI } from "@google/genai";
import path from "node:path";
import { fileURLToPath } from "node:url";

const app = express();
app.use(express.json({ limit: "12mb" }));

const project = process.env.GOOGLE_CLOUD_PROJECT || "cs663-dog-training";
const baseModel = process.env.BASE_MODEL || "gemini-3-flash-preview";
const tunedEndpoint = process.env.TUNED_ENDPOINT || "projects/917815040923/locations/us/endpoints/514008491845746688";
const apiKey = process.env.GEMINI_API_KEY || "";
const mockMode = process.env.MOCK_MODE === "true";

function locationFromEndpoint(ep) {
  const m = String(ep || "").match(/^projects\/[^/]+\/locations\/([^/]+)\//);
  return m ? m[1] : null;
}
const baseLocation = process.env.BASE_VERTEX_LOCATION || "global";
const tunedLocation = process.env.TUNED_VERTEX_LOCATION || process.env.VERTEX_LOCATION || locationFromEndpoint(tunedEndpoint) || "us";

const baseAI = apiKey
  ? new GoogleGenAI({ apiKey })
  : new GoogleGenAI({ vertexai: true, project, location: baseLocation });
const baseVia = apiKey ? "developer-api-key" : `vertex-${baseLocation}`;
const tunedAI = new GoogleGenAI({ vertexai: true, project, location: tunedLocation });

function readInput(req) {
  const body = req.body || {};
  const promptRaw = body.prompt ?? body.text;
  let imageBase64 = body.imageBase64 ?? body.image ?? "";
  const mimeType = body.mimeType || "image/jpeg";
  if (typeof promptRaw !== "string" || !promptRaw.trim()) throw new Error("prompt must be a nonempty string");
  if (typeof imageBase64 !== "string" || !imageBase64.trim()) throw new Error("imageBase64 is required");
  const dataUri = imageBase64.match(/^data:[^;]+;base64,(.+)$/s);
  if (dataUri) imageBase64 = dataUri[1];
  if (!["image/jpeg", "image/png", "image/webp"].includes(mimeType)) throw new Error("unsupported mimeType");
  return { prompt: promptRaw.trim(), imageBase64, mimeType };
}

function partsFor({ prompt, imageBase64, mimeType }) {
  return [{ text: prompt }, { inlineData: { mimeType, data: imageBase64 } }];
}

async function runModel(ai, model, input) {
  const result = await ai.models.generateContent({
    model,
    contents: [{ role: "user", parts: partsFor(input) }],
  });
  return { text: result.text || "" };
}

function mockResponse(modelName, input) {
  return {
    text: `[MOCK - not a real Gemini response] Model ${modelName} received prompt "${input.prompt}" with a ${input.mimeType} image. Set MOCK_MODE=false to get a live response.`,
    mock: true,
    model: modelName,
  };
}

async function handleBase(req, res) {
  try {
    const input = readInput(req);
    if (mockMode) return res.json(mockResponse(baseModel, input));
    res.json(await runModel(baseAI, baseModel, input));
  } catch (err) {
    console.error("base error", err);
    res.status(400).json({ error: err.message || "Base request failed" });
  }
}

async function handleTuned(req, res) {
  try {
    const input = readInput(req);
    if (mockMode) return res.json(mockResponse(tunedEndpoint, input));
    if (!tunedEndpoint) return res.status(503).json({ error: "Tuned model not configured: set TUNED_ENDPOINT." });
    res.json(await runModel(tunedAI, tunedEndpoint, input));
  } catch (err) {
    console.error("tuned error", err);
    res.status(400).json({ error: err.message || "Tuned request failed" });
  }
}

app.post("/api/gemini", handleBase);
app.post("/api/base-gemini", handleBase);
app.post("/api/fine-tuned-gemini", handleTuned);

app.get("/health", (_req, res) => res.json({
  status: "ok",
  mockMode,
  baseConfigured: true,
  baseModel,
  baseVia,
  tunedConfigured: Boolean(tunedEndpoint),
  tunedEndpoint: tunedEndpoint ? "set" : "not set",
  tunedLocation,
}));

const __dirname = path.dirname(fileURLToPath(import.meta.url));
app.use(express.static(path.join(__dirname, "public")));

const port = Number(process.env.PORT || 8080);
app.listen(port, "0.0.0.0", () => console.log(`cs663-gemini-service listening on ${port} (mockMode=${mockMode})`));
