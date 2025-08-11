# 🐤 Watermark Freedom (forked from Anything)


## License

The code and the new model trained on the [SA-1B dataset](https://ai.meta.com/datasets/segment-anything/) are under the [MIT License](LICENSE)!

> [!TIP]
> In the paper, the evaluated model was trained on the [COCO](https://cocodataset.org/#home) dataset (with additional safety filters and where faces are blurred). For reproducibility purposes, we also release the weights (see above "Weights" subsection), but this model is under the [CC-BY-NC License](LICENSE-COCO).


## Citation

If you find this repository useful, please consider giving a star :star: and please cite as:

```bibtex
@inproceedings{sander2025watermark,
  title={Watermark Anything with Localized Messages},
  author={Sander, Tom and Fernandez, Pierre and Durmus, Alain and Furon, Teddy and Douze, Matthijs},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}

```

# Local Chat Proxy (OpenAI + xAI Grok)

Expose a simple HTTP API to proxy chat requests to OpenAI or xAI (Grok), with optional streaming via SSE. Optionally, a Playwright-based browser provider can drive ChatGPT web UI for experimentation.

## Setup

1. Copy `.env.example` to `.env` and set your keys.
2. For browser provider (ChatGPT), set `CHATGPT_SESSION_TOKEN` to the `__Secure-next-auth.session-token` cookie from `chatgpt.com`.
3. Install and run in dev:

```bash
npm install
npm run dev
```

Server runs on `http://localhost:8787` by default.

## API

POST `/v1/chat`

Request body:

```json
{
  "provider": "openai" | "xai" | "browser",
  "site": "chatgpt", // required when provider is "browser"
  "model": "gpt-4o-mini",
  "messages": [{ "role": "user", "content": "Hello" }],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 512,
  "top_p": 1
}
```

- Set `stream: true` to receive Server-Sent Events with JSON lines like `{ type: "text", content: "..." }` and a final `{ type: "done" }`.

Health check: `GET /health`

## Notes

- OpenAI uses the official `openai` npm client.
- xAI Grok calls an OpenAI-compatible HTTP API at `https://api.x.ai/v1`. Adjust if your account specifies a different base URL.
- Browser provider is experimental and may break when sites change. Prefer official APIs for production.
