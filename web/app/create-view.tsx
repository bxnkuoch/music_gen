"use client";

import { useEffect, useState } from "react";

import {
  api,
  type Contexts,
  type Reference,
  type RequestCreated,
  type RequestStatus,
  requestsApi,
} from "./api";
import styles from "./page.module.css";
import SlateView from "./slate-view";

const POLL_MS = 2000;
// If nothing has finished after this long, the worker probably isn't running.
const STALL_WARNING_MS = 60_000;

type Props = {
  contexts: Contexts;
  onRated: (nRatings: number) => void;
  reference?: Reference; // generate songs that imitate this uploaded song
};

/** "Make me something": request -> progress -> ranked slate -> answer. */
export default function CreateView({ contexts, onRated, reference }: Props) {
  const [text, setText] = useState("");
  const [context, setContext] = useState("");
  const [created, setCreated] = useState<RequestCreated | null>(null);
  const [status, setStatus] = useState<RequestStatus | null>(null);
  const [startedAt, setStartedAt] = useState(0);
  const [now, setNow] = useState(0);
  const [answered, setAnswered] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const moods = Object.keys(contexts);
  const mood = context || moods[0] || "";

  useEffect(() => {
    if (!created || status?.status === "ready" || status?.status === "failed") return;
    const timer = setInterval(async () => {
      setNow(Date.now());
      try {
        setStatus(await requestsApi.status(created.request_id));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [created, status?.status]);

  async function submitRequest() {
    setError(null);
    setBusy(true);
    try {
      const result = await requestsApi.create(text, mood, reference?.reference_id ?? null);
      setCreated(result);
      setStatus(null);
      setAnswered(false);
      setStartedAt(Date.now());
      setNow(Date.now());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  async function answer(favourite: number, worst: number | null) {
    if (!status?.slate) return;
    setBusy(true);
    try {
      const result = await api.answer(status.slate.slate_id, favourite, worst);
      onRated(result.n_ratings);
      setAnswered(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  function reset() {
    setCreated(null);
    setStatus(null);
    setAnswered(false);
  }

  if (!created) {
    return (
      <section className={styles.createForm}>
        <h2>
          {reference
            ? `Describe what you want to hear, in the style of “${reference.filename}”`
            : "Describe what you want to hear"}
        </h2>
        <input
          className={styles.textInput}
          placeholder="e.g. chill lofi with plucked guitar for studying"
          value={text}
          maxLength={300}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && text.trim() && !busy && submitRequest()}
        />
        <label className={styles.muted}>
          Mood:{" "}
          <select value={mood} onChange={(e) => setContext(e.target.value)}>
            {moods.map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
        {error && <p className={styles.error}>⚠️ {error}</p>}
        <button className={styles.submit} disabled={!text.trim() || busy} onClick={submitRequest}>
          Generate 8 songs, show me the best 4
        </button>
        <p className={styles.muted}>
          Takes about 2 minutes.{" "}
          {reference
            ? "Some candidates are generated with your reference song's sound"
            : "Some candidates are nudged towards what the system has learned you like"}
          ; you won&apos;t be told which.
        </p>
      </section>
    );
  }

  const understood = [...created.tags, ...(created.energy ? [`${created.energy} energy`] : [])];
  const done = status ? status.n_done + status.n_failed : 0;
  const total = status?.n_total ?? 8;
  const stalled = done === 0 && now - startedAt > STALL_WARNING_MS;

  return (
    <section>
      <p className={styles.muted}>
        “{text}” · mood: <strong>{mood}</strong>
        {understood.length > 0 && <> · understood: {understood.join(", ")}</>}
      </p>
      {error && <p className={styles.error}>⚠️ {error}</p>}

      {(!status || status.status === "generating") && (
        <div>
          <p>
            Generating… {done}/{total} songs
          </p>
          <progress className={styles.progress} value={done} max={total} />
          {stalled && (
            <p className={styles.error}>
              Nothing has finished yet. Is the worker running? Start everything with{" "}
              <code>bash scripts/start_app.sh</code>.
            </p>
          )}
        </div>
      )}

      {status?.status === "failed" && (
        <p className={styles.error}>
          ⚠️ {status.error}{" "}
          <button className={styles.link} onClick={reset}>
            Try again
          </button>
        </p>
      )}

      {status?.status === "ready" && status.slate && !answered && (
        <SlateView key={status.slate.slate_id} slate={status.slate} busy={busy} onSubmit={answer} />
      )}

      {answered && (
        <div>
          <p>Thanks! Your choice is now part of what the system learns from.</p>
          <button className={styles.submit} onClick={reset}>
            Make another
          </button>
        </div>
      )}
    </section>
  );
}
