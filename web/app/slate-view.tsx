"use client";

import { useRef, useState } from "react";

import type { Slate } from "./api";
import styles from "./page.module.css";

// A song counts as "heard" after this many seconds of playback (seeking counts).
// Answers are only allowed once every song has been heard, to keep ratings honest.
const MIN_LISTEN_SECONDS = 5;
const LABELS = ["A", "B", "C", "D", "E", "F"];

type Props = {
  slate: Slate;
  busy: boolean;
  onSubmit: (favourite: number, worst: number | null) => void;
};

/** Play the songs in a slate and pick a favourite (and optionally a least favourite).
 *  Render with key={slate.slate_id} so its state resets for every new slate. */
export default function SlateView({ slate, busy, onSubmit }: Props) {
  const [heard, setHeard] = useState<Set<number>>(new Set());
  const [favourite, setFavourite] = useState<number | null>(null);
  const [worst, setWorst] = useState<number | null>(null);
  const players = useRef<Map<number, HTMLAudioElement>>(new Map());

  function onPlay(songId: number) {
    // Only one song plays at a time.
    players.current.forEach((player, id) => id !== songId && player.pause());
  }

  function onTimeUpdate(songId: number, seconds: number) {
    if (seconds >= MIN_LISTEN_SECONDS && !heard.has(songId)) {
      setHeard((prev) => new Set(prev).add(songId));
    }
  }

  const allHeard = slate.songs.every((s) => heard.has(s.song_id));

  return (
    <section>
      <p className={styles.muted}>
        Listen to each song for at least {MIN_LISTEN_SECONDS}s (skipping ahead counts), then
        pick your favourite and, if you like, your least favourite.
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
                  name={`favourite-${slate.slate_id}`}
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
        onClick={() => favourite !== null && onSubmit(favourite, worst)}
      >
        {busy ? "Saving…" : allHeard ? "Submit" : "Listen to every song first"}
      </button>
    </section>
  );
}
