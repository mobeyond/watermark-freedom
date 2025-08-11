"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.config = void 0;
exports.assertEnv = assertEnv;
const dotenv_1 = __importDefault(require("dotenv"));
dotenv_1.default.config();
exports.config = {
    port: Number(process.env.PORT || 8787),
    openaiApiKey: process.env.OPENAI_API_KEY || "",
    xaiApiKey: process.env.XAI_API_KEY || "",
    corsOrigin: process.env.CORS_ORIGIN || "*",
};
function assertEnv() {
    // Nothing mandatory at startup; provider-specific checks happen on use
    return true;
}
//# sourceMappingURL=config.js.map