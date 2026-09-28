"use client";

import { useCallback, useEffect, useState } from "react";

import { api, type Contexts, type Profile, type Slate } from "./api";
import CreateView from "./create-view";
import styles from "./page.module.css";
import ReferenceView from "./reference-view";
import SlateView from "./slate-view";

type Tab = "rate" | "create" | "reference";

export default function RatingApp() {
  const [tab, setTab] = useState<Tab>("rate");
  const [contexts, setContexts] = useState<Contexts>({});
  const [session, setSession] = useState<{ id: number; context: string } | null>(null);
  const [slate, setSlate] = useState<Slate | null>(null);
  const [nRatings, setNRatings] = useState(0);
  const [profile, setProfile] = useState<Profile | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const report = (e: unknown) => setError(e instanceof Error ? e.message : String(e));

  useEffect(() => {
    api.contexts().then(setContexts).catch(report);
    api.stats().then((s) => setNRatings(s.n_ratings)).catch(report);
    api.profile().then(setProfile).catch(report);
  }, []);

  const loadSlate = useCallback(async (sessionId: number) => {
    setSlate(await api.nextSlate(sessionId));
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

  async function onRated(n: number) {
    setNRatings(n);
    if (n % 5 === 0) setProfile(await api.profile());
  }

  async function submit(favourite: number, worst: number | null) {
    if (!slate || !session) return;
    setBusy(true);
    setError(null);
    try {
      const result = await api.answer(slate.slate_id, favourite, worst);
      await onRated(result.n_ratings);
      await loadSlate(session.id);
    } catch (e) {
      report(e);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <nav className={styles.tabs}>
          <button className={tab === "rate" ? styles.tabActive : styles.tab} onClick={() => setTab("rate")}>
            Rate library
          </button>
          <button
            className={tab === "create" ? styles.tabActive : styles.tab}
            onClick={() => setTab("create")}
          >
            Create new
          </button>
          <button
            className={tab === "reference" ? styles.tabActive : styles.tab}
            onClick={() => setTab("reference")}
          >
            More like this
          </button>
        </nav>
        <p className={styles.muted}>
          {nRatings} rating{nRatings === 1 ? "" : "s"} so far
          {tab === "rate" && session && (
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

      {tab === "rate" && !session && (
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

      {tab === "rate" && session && slate && (
        <SlateView key={slate.slate_id} slate={slate} busy={busy} onSubmit={submit} />
      )}

      {tab === "create" && <CreateView contexts={contexts} onRated={onRated} />}
      {tab === "reference" && <ReferenceView contexts={contexts} onRated={onRated} />}

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
