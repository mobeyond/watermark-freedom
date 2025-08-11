"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.XAIProvider = void 0;
const axios_1 = __importDefault(require("axios"));
const config_1 = require("../config");
class XAIProvider {
    constructor(options) {
        if (!config_1.config.xaiApiKey) {
            throw new Error("XAI_API_KEY is required to use the xAI provider");
        }
        this.apiKey = config_1.config.xaiApiKey;
        // xAI aims to be OpenAI-compatible; adjust if your account docs show a different base URL
        this.baseURL = options?.baseURL || "https://api.x.ai/v1";
    }
    async chat(params) {
        const { model = "grok-2-latest", messages, stream = false, temperature, max_tokens, top_p, } = params;
        const payload = {
            model,
            messages,
            stream,
            temperature,
            max_tokens,
            top_p,
        };
        const headers = {
            Authorization: `Bearer ${this.apiKey}`,
            "Content-Type": "application/json",
            Accept: stream ? "text/event-stream" : "application/json",
        };
        if (stream) {
            const response = await axios_1.default.post(`${this.baseURL}/chat/completions`, payload, {
                headers,
                responseType: "stream",
            });
            return response.data;
        }
        const response = await axios_1.default.post(`${this.baseURL}/chat/completions`, payload, { headers });
        const text = response.data?.choices?.[0]?.message?.content ?? "";
        return { text };
    }
}
exports.XAIProvider = XAIProvider;
//# sourceMappingURL=xai.js.map