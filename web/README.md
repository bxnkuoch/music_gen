# Rating page (Next.js + TypeScript)

The browser side of the rating loop. It talks to the Python API through the
`/api/*` proxy in `next.config.ts` (default target: http://localhost:8000).

Start everything from the repo root with `bash scripts/start_app.sh`, or just this
part with `npm run dev` (the API must already be running).

- `app/rating-app.tsx`: the page (mood picker → slate → answer → next)
- `app/api.ts`: typed calls to the Python API
- `app/page.module.css`: styles

Checks: `npm run lint && npx tsc --noEmit`
