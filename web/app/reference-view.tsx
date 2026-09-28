"use client";

import { useEffect, useState } from "react";

import { type Contexts, type Reference, referencesApi } from "./api";
import CreateView from "./create-view";
import styles from "./page.module.css";

const POLL_MS = 1000;
const MAX_MB = 50; // the API's limit (references.MAX_BYTES)

type Props = { contexts: Contexts; onRated: (nRatings: number) => void };

/** "More like this": upload a song -> closest library songs -> generate in its style. */
export default function ReferenceView({ contexts, onRated }: Props) {
  const [reference, setReference] = useState<Reference | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (reference?.status !== "pending") return;
    const timer = setInterval(async () => {
      try {
        setReference(await referencesApi.get(reference.reference_id));
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    }, POLL_MS);
    return () => clearInterval(timer);
  }, [reference?.reference_id, reference?.status]);

  async function upload(file: File) {
    if (file.size > MAX_MB * 2 ** 20) {
      setError(`That file is larger than ${MAX_MB} MB.`);
      return;
    }
    setError(null);
    setBusy(true);
    try {
      setReference(await referencesApi.upload(file));
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (!reference) {
    return (
      <section className={styles.createForm}>
        <h2>Start from a song you like</h2>
        <input
          type="file"
          accept=".mp3,.wav,.flac,.ogg"
          disabled={busy}
          onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])}
        />
        {busy && <p className={styles.muted}>Uploading…</p>}
        {error && <p className={styles.error}>⚠️ {error}</p>}
        <p className={styles.muted}>
          MP3, WAV, FLAC or OGG, up to {MAX_MB} MB. The file is kept on this computer only (in{" "}
          <code>data/</code>), so it can be replayed and used to generate from.
        </p>
      </section>
    );
  }

  return (
    <section>
      <p className={styles.muted}>
        Reference: <strong>{reference.filename}</strong> ({Math.round(reference.duration_s)} s){" "}
        <button className={styles.link} onClick={() => setReference(null)}>
          change
        </button>
      </p>
      <audio controls preload="none" src={reference.audio_url} />
      {error && <p className={styles.error}>⚠️ {error}</p>}

      {reference.status === "pending" && (
        <p className={styles.muted}>Listening to it… (the worker must be running)</p>
      )}
      {reference.status === "failed" && (
        <p className={styles.error}>⚠️ Couldn&apos;t analyse this song: {reference.error}</p>
      )}
      {reference.status === "ready" && (
        <>
          <h2>Closest songs in your library</h2>
          <ol className={styles.slate}>
            {reference.similar.map((song) => (
              <li key={song.song_id} className={styles.card}>
                <div className={styles.cardTop}>
                  <span className={styles.muted}>{song.prompt}</span>
                  <span className={styles.muted}>
                    similarity {(1 - song.distance).toFixed(2)}
                  </span>
                </div>
                <audio controls preload="none" src={song.audio_url} />
              </li>
            ))}
          </ol>
        </>
      )}

      {reference.status !== "failed" && (
        <CreateView
          key={reference.reference_id}
          contexts={contexts}
          onRated={onRated}
          reference={reference}
        />
      )}
    </section>
  );
}
