"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.OpenAIProvider = void 0;
const openai_1 = __importDefault(require("openai"));
const config_1 = require("../config");
class OpenAIProvider {
    constructor() {
        if (!config_1.config.openaiApiKey) {
            throw new Error("OPENAI_API_KEY is required to use the OpenAI provider");
        }
        this.client = new openai_1.default({ apiKey: config_1.config.openaiApiKey });
    }
    async chat(params) {
        const { model = "gpt-4o-mini", messages, stream = false, temperature, max_tokens = null, top_p, } = params;
        const openaiMessages = messages.map((m) => ({ role: m.role, content: m.content }));
        const base = {
            model,
            messages: openaiMessages,
        };
        if (typeof temperature === "number")
            base.temperature = temperature;
        if (max_tokens !== null)
            base.max_tokens = max_tokens;
        if (typeof top_p === "number")
            base.top_p = top_p;
        if (stream) {
            const streamResp = await this.client.chat.completions.create({
                ...base,
                stream: true,
            });
            return streamResp;
        }
        const resp = await this.client.chat.completions.create(base);
        const text = resp.choices?.[0]?.message?.content ?? "";
        return { text };
    }
}
exports.OpenAIProvider = OpenAIProvider;
//# sourceMappingURL=openai.js.map