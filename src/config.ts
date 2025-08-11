import dotenv from "dotenv";

dotenv.config();

export const config = {
  port: Number(process.env.PORT || 8787),
  openaiApiKey: process.env.OPENAI_API_KEY || "",
  xaiApiKey: process.env.XAI_API_KEY || "",
  corsOrigin: process.env.CORS_ORIGIN || "*",
};

export function assertEnv() {
  // Nothing mandatory at startup; provider-specific checks happen on use
  return true;
}