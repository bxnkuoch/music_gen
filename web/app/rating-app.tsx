"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api, type Contexts, type Profile, type Slate } from "./api";
import styles from "./page.module.css";

// A song counts as "heard" after this many seconds of playback. Answers are only
// allowed once every song in the slate has been heard, to keep ratings honest.
const MIN_LISTEN_SECONDS = 5;
const LABELS = ["A", "B", "C", "D", "E", "F"];

export default function RatingApp() {
  const [contexts, setContexts] = useState<Contexts>({});
  const [session, setSession] = useState<{ id: number; context: string } | null>(null);
  const [slate, setSlate] = useState<Slate | null>(null);
  const [heard, setHeard] = useState<Set<number>>(new Set());
  const [favourite, setFavourite] = useState<number | null>(null);
  const [worst, setWorst] = useState<number | null>(null);
  const [nRatings, setNRatings] = useState(0);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const players = useRef<Map<number, HTMLAudioElement>>(new Map());

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  useEffect(() => {
    api.contexts().then(setContexts).catch(report);
    api.stats().then((s) => setNRatings(s.n_ratings)).catch(report);
    api.profile().then(setProfile).catch(report);
  }, []);

  const loadSlate = useCallback(async (sessionId: number) => {
    setSlate(await api.nextSlate(sessionId));
    setHeard(new Set());
    setFavourite(null);
    setWorst(null);
  }, []);

  async function chooseMood(context: string) {
    setError(null);
    try {
      const s = await api.startSession(context);
      setSession({ id: s.session_id, context: s.context });
      await loadSlate(s.session_id);
    } catch (e) {
      report(e);
    }
  }

  async function submit() {
    if (!slate || !session || favourite === null) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.answer(slate.slate_id, favourite, worst);
      setNRatings(result.n_ratings);
      if (result.n_ratings % 5 === 0) setProfile(await api.profile());
      await loadSlate(session.id);
    } catch (e) {
      report(e);
    } finally {
      setBusy(false);
    }
  }

  function onPlay(songId: number) {
    // Only one song plays at a time.
    players.current.forEach((player, id) => id !== songId && player.pause());
  }

  function onTimeUpdate(songId: number, seconds: number) {
    if (seconds >= MIN_LISTEN_SECONDS && !heard.has(songId)) {
      setHeard((prev) => new Set(prev).add(songId));
    }
  }

  const allHeard = slate !== null && slate.songs.every((s) => heard.has(s.song_id));

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1>Which one do you like best?</h1>
        <p className={styles.muted}>
          {nRatings} rating{nRatings === 1 ? "" : "s"} so far
          {session && (
            <>
              {" · mood: "}
              <strong>{session.context}</strong>{" "}
              <button className={styles.link} onClick={() => setSession(null)}>
                change
              </button>
            </>
          )}
        </p>
      </header>

      {error && <p className={styles.error}>⚠️ {error}</p>}

      {!session && (
        <section>
          <h2>What mood are you in?</h2>
          <div className={styles.moods}>
            {Object.entries(contexts).map(([name, description]) => (
              <button key={name} className={styles.mood} onClick={() => chooseMood(name)}>
                <strong>{name}</strong>
                <span>{description}</span>
              </button>
            ))}
          </div>
        </section>
      )}

      {session && slate && (
        <section>
          <p className={styles.muted}>
            Listen to each song for at least {MIN_LISTEN_SECONDS}s, pick your favourite
            (and, if you like, your least favourite).
          </p>
          <ol className={styles.slate}>
            {slate.songs.map((song, i) => (
              <li key={song.song_id} className={styles.card}>
                <div className={styles.cardTop}>
                  <span className={styles.label}>{LABELS[i]}</span>
                  {heard.has(song.song_id) ? "✓ heard" : "not heard yet"}
                </div>
                <audio
                  controls
                  preload="none"
                  src={song.audio_url}
                  ref={(el) => {
                    if (el) players.current.set(song.song_id, el);
                    else players.current.delete(song.song_id);
                  }}
                  onPlay={() => onPlay(song.song_id)}
                  onTimeUpdate={(e) => onTimeUpdate(song.song_id, e.currentTarget.currentTime)}
                />
                <div className={styles.choices}>
                  <label>
                    <input
                      type="radio"
                      name="favourite"
                      checked={favourite === song.song_id}
                      onChange={() => {
                        setFavourite(song.song_id);
                        if (worst === song.song_id) setWorst(null);
                      }}
                    />{" "}
                    Favourite
                  </label>
                  <label>
                    <input
                      type="checkbox"
                      checked={worst === song.song_id}
                      disabled={favourite === song.song_id}
                      onChange={(e) => setWorst(e.target.checked ? song.song_id : null)}
                    />{" "}
                    Least favourite
                  </label>
                </div>
              </li>
            ))}
          </ol>
          <button
            className={styles.submit}
            disabled={!allHeard || favourite === null || busy}
            onClick={submit}
          >
            {busy ? "Saving…" : allHeard ? "Submit & next" : "Listen to every song first"}
          </button>
        </section>
      )}

      {profile && profile.n_ratings > 0 && (
        <aside className={styles.profile}>
          <h2>What it has learned about you</h2>
          <p className={styles.muted}>
            Based on {profile.n_ratings} ratings. Updates every 5 ratings. Early on this is
            mostly noise.
          </p>
          <div className={styles.columns}>
            <FeatureList title="Leans towards" features={profile.likes} />
            <FeatureList title="Leans away from" features={profile.dislikes} />
          </div>
        </aside>
      )}
    </main>
  );
}

function FeatureList({ title, features }: { title: string; features: Profile["likes"] }) {
  return (
    <div>
      <h3>{title}</h3>
      {features.length === 0 ? (
        <p className={styles.muted}>nothing yet</p>
      ) : (
        <ul>
          {features.map((f) => (
            <li key={f.feature}>{f.feature.replaceAll("_", " ")}</li>
          ))}
        </ul>
      )}
    </div>
  );
}
