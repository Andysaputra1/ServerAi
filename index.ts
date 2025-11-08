import express from "express";
import cors from "cors";
import "dotenv/config";
import fs from "node:fs";
import path from "node:path";
import OpenAI from "openai";

const app = express();
app.use(cors());
app.use(express.json());

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

type Chunk = { source: string; text: string; embedding: number[]; score?: number };
let STORE: Chunk[] = [];

const MODEL_EMB = "text-embedding-3-large";
const MODEL_CHAT = "gpt-4.1-mini";
const TOP_K = 6;

// path.join yang aman untuk Windows/Linux
function safeJoin(...parts: string[]) {
  return path.join(...parts).replace(/\\/g, "/");
}

// Gunakan __dirname untuk path yang reliabel setelah kompilasi TS
const DATA_PATH = safeJoin(__dirname, "..", "data", "profile.json");
const CACHE_PATH = safeJoin(__dirname, "..", "data", "store.cache.json");

// ---------- utils
function chunkText(text: string, size = 800, overlap = 120) {
  const words = text.split(/\s+/);
  const out: string[] = [];
  for (let i = 0; i < words.length; i += (size - overlap)) {
    out.push(words.slice(i, i + size).join(" "));
    if (i + size >= words.length) break;
  }
  return out.filter(Boolean);
}
function cosine(a: number[], b: number[]) {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) { dot += a[i]*b[i]; na += a[i]*a[i]; nb += b[i]*b[i]; }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
}
async function embedOne(text: string) {
  try {
    const e = await openai.embeddings.create({ model: MODEL_EMB, input: text });
    return e.data[0].embedding as number[];
  } catch (err: any) {
    console.error(`[embedOne Error] ${err.message}`);
    // Kembalikan array kosong atau throw error, tergantung kebutuhan
    // Untuk saat ini, kita kembalikan array nol agar tidak gagal total
    // Sebaiknya, Anda perlu menangani embedding yang gagal dengan lebih baik
    return new Array(3072).fill(0); // Sesuaikan dimensi model jika 'text-embedding-3-large' berubah
  }
}

// ---------- [BARU] Fungsi build yang di-refactor
/**
 * Membaca profile.json, mengubahnya menjadi text entries,
 * men-chunk, dan memanggil API OpenAI untuk embedding.
 */
async function createAndEmbedEntries(): Promise<Chunk[]> {
  const raw = fs.readFileSync(DATA_PATH, "utf8");
  const data = JSON.parse(raw);

  // Normalisasi key
  if (data.profile?.["Soft_Skills:"]) {
    data.profile.soft_skills = data.profile["Soft_Skills:"];
    delete data.profile["Soft_Skills:"];
  }
  if (data.profile?.Hard_Skills) {
    data.profile.hard_skills = data.profile.Hard_Skills;
    delete data.profile.Hard_Skills;
  }
  if (data.profile?.Language) {
    data.profile.languages = data.profile.Language;
    delete data.profile.Language;
  }
  // [PERBAIKAN] Ambil Achievement
  if (data.profile?.Achievement) {
    data.profile.achievement = data.profile.Achievement;
    delete data.profile.Achievement;
  }
  // Normalisasi key 'Education'
  const education = data.education ?? data.Education ?? [];
  
  const entries: { source: string; text: string }[] = [];

  // PROFILE
  if (data.profile?.summary) entries.push({ source: "profile:summary", text: String(data.profile.summary) });
  if (data.profile?.full_name || data.profile?.headline) {
    entries.push({
      source: "profile:headline",
      text: `${data.profile.full_name ?? ""}\n${data.profile.headline ?? ""}`.trim()
    });
  }
  // [PERBAIKAN] Tambahkan Achievement
  if (data.profile?.achievement) {
    entries.push({ source: "profile:achievement", text: `Achievement: ${data.profile.achievement}`});
  }
  if (data.profile?.hard_skills || data.profile?.soft_skills || data.profile?.languages) {
    entries.push({
      source: "profile:skills",
      text: [
        data.profile.hard_skills ? `Hard skills: ${data.profile.hard_skills}` : "",
        data.profile.soft_skills ? `Soft skills: ${data.profile.soft_skills}` : "",
        data.profile.languages   ? `Languages: ${data.profile.languages}`   : "",
      ].filter(Boolean).join("\n")
    });
  }

  // PROJECTS
  for (const prj of (data.projects || [])) {
    entries.push({
      source: `project:${prj.name}`,
      text: [
        prj.name, prj.description,
        prj.impact ? `Impact: ${prj.impact}` : "",
        (prj.tech_stack && prj.tech_stack.length) ? `Stack: ${prj.tech_stack.join(", ")}` : "",
        prj.repo_url ?? ""
      ].filter(Boolean).join("\n")
    });
  }

  // EXPERIENCES
  for (const exp of (data.experiences || [])) {
    const org = exp.organization || "Unknown";
    const bullets = exp.highlights || exp.responsibilities || [];
    const period = `${exp.start_date ?? ""}–${(exp.end_date ?? "Present")}`;
    const extra = Array.isArray(exp.projects) && exp.projects.length ? `Projects: ${exp.projects.join(", ")}` : "";
    entries.push({
      source: `exp:${org}`,
      text: [`${exp.role} @ ${org} (${period})`, bullets.join("; "), extra].filter(Boolean).join("\n")
    });
  }

  // EDUCATION
  for (const ed of (education || [])) {
    const period = `${ed.start_date ?? ""}–${(ed.end_date_expected ?? ed.end_date ?? "")}`;
    const coursework = Array.isArray(ed.coursework) && ed.coursework.length ? `Coursework: ${ed.coursework.join(", ")}` : "";
    const activities = Array.isArray(ed.activities) && ed.activities.length ? `Activities: ${ed.activities.join(", ")}` : "";
    const campusOrLoc = ed.campus ? `Campus: ${ed.campus}` : (ed.location ? `Location: ${ed.location}` : "");
    const gpa = ed.gpa ? `GPA: ${ed.gpa}` : "";
    entries.push({
      source: `edu:${ed.institution}`,
      text: [
        `${ed.institution}${campusOrLoc ? `, ${campusOrLoc}` : ""}`,
        [ed.degree ?? ed.program ?? "", ed.stream ? `(${ed.stream})` : ""].join(" ").trim(),
        period, gpa, coursework, activities
      ].filter(Boolean).join("\n")
    });
  }

  // FAQs
  for (const f of (data.faqs || [])) {
    entries.push({ source: `faq:${f.q}`, text: `${f.q}\n${f.a}` });
  }

  // CHUNK + EMBED
  const chunks: { source: string; text: string }[] = [];
  for (const e of entries) for (const c of chunkText(e.text)) chunks.push({ source: e.source, text: c });

  console.log(`[build] Menerima ${entries.length} entri, dipecah menjadi ${chunks.length} chunks.`);
  console.log(`[build] Memulai proses embedding... (ini mungkin perlu waktu)`);

  // [PERBAIKAN] Gunakan Promise.all untuk embedding paralel
  const embeddingPromises = chunks.map(c => embedOne(c.text));
  const embeddings = await Promise.all(embeddingPromises);

  const batched: Chunk[] = chunks.map((c, i) => ({
    source: c.source,
    text: c.text,
    embedding: embeddings[i],
  }));

  console.log(`[build] Selesai embedding ${batched.length} chunks.`);
  return batched.filter(b => b.embedding && b.embedding.length > 0); // Filter yg gagal
}

// ---------- [BARU] Fungsi buildStore dengan Caching
async function buildStore() {
  // Tambahkan 'REBUILD_STORE=true' di .env jika Anda ingin memaksa build ulang
  const forceRebuild = process.env.REBUILD_STORE === 'true';
  
  if (!forceRebuild && fs.existsSync(CACHE_PATH)) {
    console.log(`[build] Memuat cache store dari ${CACHE_PATH}...`);
    const rawCache = fs.readFileSync(CACHE_PATH, "utf8");
    STORE = JSON.parse(rawCache);
    console.log(`[build] Cache store dimuat. ${STORE.length} chunks.`);
    return;
  }

  console.log(`[build] Cache tidak ditemukan atau REBUILD_STORE=true. Membangun store baru...`);
  try {
    const newStore = await createAndEmbedEntries();
    fs.writeFileSync(CACHE_PATH, JSON.stringify(newStore));
    console.log(`[build] Store baru disimpan ke cache: ${CACHE_PATH}`);
    STORE = newStore;
  } catch (err: any) {
    console.error(`[build] Gagal membangun store: ${err.message}`);
    process.exit(1); // Gagal total jika build pertama gagal
  }
}

const SYSTEM_PROMPT = `
You are the candidate’s public career chatbot. Audience: HR and general public.
Tone: professional, concise, friendly. Mirror Indonesian/English automatically.
Answer ONLY from the provided context. If missing, say you don't have that info.
Refuse sensitive PII (NIK/NPWP/SSN, full home address, family, religion, marital status).
Do NOT state salary unless present in context.
Always include short tags derived from the context you used (e.g., ).
`;

// ---------- API
app.get("/api/health", (_req, res) => res.json({ ok: true, chunks: STORE.length }));

app.get("/api/debug/store", (_req, res) => {
  const bySrc = STORE.reduce((m, c) => {
    m[c.source] = (m[c.source] || 0) + 1;
    return m;
  }, {} as Record<string, number>);
  res.json({ chunks: STORE.length, bySrc });
});

app.post("/api/chat", async (req, res) => {
  try {
    const question: string = req.body?.question ?? "";
    if (!question) return res.status(400).json({ error: "question required" });

    if (STORE.length === 0) {
      return res.status(503).json({ error: "Vector store not initialized yet. Please wait." });
    }

    const qEmb = await embedOne(question);
    const ranked = STORE
      .map((c) => ({ ...c, score: cosine(qEmb, c.embedding) }))
      .sort((a, b) => (b.score! - a.score!))
      .slice(0, TOP_K);

    const context = ranked.map(r => `SOURCE: ${r.source}\n${r.text}`).join("\n\n---\n\n");

    const cmp = await openai.chat.completions.create({
      model: MODEL_CHAT,
      temperature: 0.2,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: `Context:\n${context}\n\nQuestion:\n${question}\n\nAnswer with tags.` }
      ]
    });

    res.json({ answer: cmp.choices?.[0]?.message?.content ?? "No answer." });
  } catch (e: any) {
    console.error(e);
    res.status(500).json({ error: e?.message || "server error" });
  }
});

// ---------- start
buildStore().then(() => {
  app.get("/", (_req, res) => res.send(`OK. Store loaded with ${STORE.length} chunks.`));
  const PORT = Number(process.env.PORT) || 8787;
  app.listen(PORT, () => console.log(`✅ API running on port ${PORT}`));
}).catch(err => {
  console.error("Failed to initialize:", err);
  process.exit(1);
});