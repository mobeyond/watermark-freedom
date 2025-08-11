"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
const express_1 = __importDefault(require("express"));
const cors_1 = __importDefault(require("cors"));
const config_1 = require("./config");
const openai_1 = require("./providers/openai");
const xai_1 = require("./providers/xai");
const browser_1 = require("./providers/browser");
const app = (0, express_1.default)();
app.use(express_1.default.json({ limit: "2mb" }));
app.use((0, cors_1.default)({
    origin: (origin, callback) => callback(null, config_1.config.corsOrigin === "*" ? true : config_1.config.corsOrigin),
    credentials: true,
}));
app.get("/health", (_req, res) => {
    res.json({ ok: true });
});
function getProvider(providerId) {
    switch (providerId) {
        case "openai":
            return new openai_1.OpenAIProvider();
        case "xai":
            return new xai_1.XAIProvider();
        case "browser":
            return new browser_1.BrowserProvider();
        default:
            throw new Error(`Unsupported provider: ${providerId}`);
    }
}
app.post("/v1/chat", async (req, res) => {
    const body = req.body;
    if (!body || !body.provider || !Array.isArray(body.messages)) {
        return res.status(400).json({ error: "Invalid request body" });
    }
    try {
        const provider = getProvider(body.provider);
        if (body.stream) {
            res.setHeader("Content-Type", "text/event-stream");
            res.setHeader("Cache-Control", "no-cache, no-transform");
            res.setHeader("Connection", "keep-alive");
            const flush = (data) => {
                res.write(`data: ${data}\n\n`);
            };
            const end = () => res.end();
            const providerResp = await provider.chat({
                model: body.model,
                site: body.site,
                messages: body.messages,
                stream: true,
                temperature: body.temperature,
                max_tokens: body.max_tokens,
                top_p: body.top_p,
            });
            if (providerResp && typeof providerResp[Symbol.asyncIterator] === "function") {
                for await (const delta of providerResp) {
                    const content = delta?.choices?.[0]?.delta?.content;
                    if (typeof content === "string" && content.length > 0) {
                        flush(JSON.stringify({ type: "text", content }));
                    }
                }
                flush(JSON.stringify({ type: "done" }));
                end();
                return;
            }
            if (providerResp && typeof providerResp.on === "function") {
                providerResp.on("data", (chunk) => {
                    const s = chunk.toString("utf8");
                    s.split(/\r?\n/).forEach((line) => {
                        if (!line.startsWith("data:"))
                            return;
                        const json = line.slice(5).trim();
                        if (json === "[DONE]") {
                            flush(JSON.stringify({ type: "done" }));
                        }
                        else if (json) {
                            try {
                                const obj = JSON.parse(json);
                                const content = obj?.choices?.[0]?.delta?.content;
                                if (typeof content === "string" && content.length > 0) {
                                    flush(JSON.stringify({ type: "text", content }));
                                }
                            }
                            catch {
                            }
                        }
                    });
                });
                providerResp.on("end", () => {
                    flush(JSON.stringify({ type: "done" }));
                    end();
                });
                providerResp.on("error", (err) => {
                    flush(JSON.stringify({ type: "error", error: err.message }));
                    end();
                });
                return;
            }
            flush(JSON.stringify({ type: "error", error: "Unknown stream response" }));
            end();
            return;
        }
        const result = await provider.chat({
            model: body.model,
            site: body.site,
            messages: body.messages,
            temperature: body.temperature,
            max_tokens: body.max_tokens,
            top_p: body.top_p,
            stream: false,
        });
        return res.json({ content: result.text || "" });
    }
    catch (err) {
        const status = err?.response?.status || 500;
        const message = err?.response?.data || err?.message || "Internal error";
        return res.status(status).json({ error: message });
    }
});
app.listen(config_1.config.port, () => {
    // eslint-disable-next-line no-console
    console.log(`Chat proxy listening on http://localhost:${config_1.config.port}`);
});
//# sourceMappingURL=server.js.map