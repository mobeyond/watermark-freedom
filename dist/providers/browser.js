"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.BrowserProvider = void 0;
const playwright_1 = require("playwright");
class BrowserProvider {
    async chat(params) {
        const { site, messages, stream = false } = params;
        if (site === "chatgpt") {
            if (!process.env.CHATGPT_SESSION_TOKEN) {
                throw new Error("CHATGPT_SESSION_TOKEN env var is required for browser provider (ChatGPT)");
            }
            const result = await this.chatWithChatGPT({ messages, stream });
            return result;
        }
        if (site === "grok") {
            throw new Error("Browser automation for Grok is not implemented yet. Use provider=\"xai\".");
        }
        throw new Error(`Unsupported browser site: ${site}`);
    }
    async chatWithChatGPT(params) {
        const { messages, stream } = params;
        const latestUser = this.buildPrompt(messages);
        const browser = await playwright_1.chromium.launch({ headless: true, args: ["--no-sandbox", "--disable-setuid-sandbox"] });
        let context;
        let page;
        try {
            context = await browser.newContext();
            // Set session cookie
            await context.addCookies([
                {
                    name: "__Secure-next-auth.session-token",
                    value: process.env.CHATGPT_SESSION_TOKEN,
                    domain: ".chatgpt.com",
                    path: "/",
                    httpOnly: true,
                    secure: true,
                    sameSite: "Lax",
                },
            ]);
            page = await context.newPage();
            await page.goto("https://chatgpt.com/", { waitUntil: "domcontentloaded" });
            // Wait for prompt area
            await page.waitForSelector('[data-testid="prompt-textarea"]', { timeout: 30000 });
            await page.click('[data-testid="prompt-textarea"]');
            await page.keyboard.type(latestUser, { delay: 10 });
            // Send
            await page.click('[data-testid="send-button"]');
            if (stream) {
                return this.streamAssistantText(page);
            }
            const text = await this.waitForAssistantFinalText(page);
            return { text };
        }
        finally {
            // Close with slight delay to allow streaming consumer to attach
            if (!stream) {
                await page?.close().catch(() => { });
                await context?.close().catch(() => { });
                await browser.close().catch(() => { });
            }
        }
    }
    buildPrompt(messages) {
        // Simple approach: include system priming and last user content
        const system = messages.find((m) => m.role === "system")?.content;
        const lastUser = [...messages].reverse().find((m) => m.role === "user")?.content || "";
        if (system) {
            return `System: ${system}\n\nUser: ${lastUser}`;
        }
        return lastUser;
    }
    async streamAssistantText(page) {
        // Return a Node stream-like interface using an async generator to fit server implementation
        // We create an async generator that yields chunks as the assistant text grows
        async function* generator() {
            let last = "";
            let stableIterations = 0;
            while (true) {
                const current = await BrowserProvider.readLastAssistantText(page);
                if (current && current.length > last.length) {
                    const delta = current.slice(last.length);
                    last = current;
                    yield { choices: [{ delta: { content: delta } }] };
                    stableIterations = 0;
                }
                else {
                    stableIterations += 1;
                }
                // Heuristic: if stable for ~2 seconds, finish
                if (stableIterations >= 8) {
                    break;
                }
                await page.waitForTimeout(250);
            }
        }
        // Return the async iterable
        return generator();
    }
    async waitForAssistantFinalText(page) {
        let last = "";
        let stableIterations = 0;
        for (let i = 0; i < 240; i++) {
            const current = await BrowserProvider.readLastAssistantText(page);
            if (current && current.length > last.length) {
                last = current;
                stableIterations = 0;
            }
            else {
                stableIterations += 1;
            }
            if (stableIterations >= 8) {
                break;
            }
            await page.waitForTimeout(250);
        }
        return last;
    }
    static async readLastAssistantText(page) {
        return page.evaluate(() => {
            function getAssistantText() {
                // Try various selectors for assistant messages
                const assistantRoles = Array.from(document.querySelectorAll('[data-message-author-role="assistant"], [data-testid^="conversation-turn-"]'));
                const nodes = assistantRoles.filter((el) => el.textContent && el.textContent.trim().length > 0);
                const last = nodes[nodes.length - 1];
                return last?.innerText?.trim() || "";
            }
            return getAssistantText();
        });
    }
}
exports.BrowserProvider = BrowserProvider;
//# sourceMappingURL=browser.js.map