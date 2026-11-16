# How to run zypherLL

The web chat needs two things running: the **API** (it loads the model) and
the **web app** (the chat page in your browser).

## One-time setup

Do this once per machine.

1. Install the Python dependencies:
   ```bash
   cd D:\GIT\zypherLL
   pip install -r requirements.txt
   ```

2. Create the API settings file, if you have not already:
   ```bash
   copy .env.example .env
   ```
   Open `.env` and set `ZYPHER_API_KEY` (any long random value) and
   `ZYPHER_MODEL_PATH` (where the model is stored). To make a key:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. Install the web app:
   ```bash
   cd D:\GIT\zypherLL\web
   npm install
   copy .env.example .env
   ```
   Open `web\.env` and set `ZYPHER_API_KEY` to the **same value** as in
   `D:\GIT\zypherLL\.env`.

The first start downloads the model (4.4 GB), llama.cpp (~1.1 GB) and the
memory encoder (90 MB). Later starts use what is already on disk.

## Each time

Open two terminals, in this order.

1. **Start the API** (it loads the model):
   ```bash
   cd D:\GIT\zypherLL
   python main.py serve
   ```
   Wait for the model to load. The first start after a reboot takes 1-2
   minutes (reading 4.4 GB off the hard disk); later starts take seconds.

2. **Start the web app:**
   ```bash
   cd D:\GIT\zypherLL\web
   npm run dev
   ```

3. **Open http://127.0.0.1:5173** in your browser.
   - Green dot next to `zephyr-7b`: ready.
   - Yellow banner: the model is still loading.
   - Red banner: the API is not running.

## Stop

Press Ctrl+C in both terminals. Stopping the API frees the GPU memory.

## Optional: production build

Faster page load, same setup. Instead of step 2:

```bash
cd D:\GIT\zypherLL\web
npm run build
npm run preview
```

Then open http://127.0.0.1:4173. Run `npm run build` again after changing the
web app's code.

## Console chat instead of the browser

No web app needed, one terminal:

```bash
cd D:\GIT\zypherLL
python main.py
```

## If it doesn't work

| Problem | Fix |
|---|---|
| Red "API is not running" banner | Step 1 is not running or crashed. Check its terminal |
| 401 / "invalid or missing API key" | The keys in `.env` and `web\.env` do not match |
| Answers slow down mid-chat | The GPU is overheating; a notice shows under the answer. Give the laptop airflow |
| Slow from the first answer | Another program is using GPU memory. Close it and restart the API |
| `npm` not found | Install Node.js 20.19 or newer |

More detail: `README.md` (the model and API), `web/README.md` (the web app),
`projectDetail.md` (architecture and full API reference).
