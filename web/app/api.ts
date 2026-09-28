// Typed wrappers around the Python API (see src/music_gen/api/app.py).

export type Contexts = Record<string, string>;
export type SlateSong = { song_id: number; audio_url: string };
export type Slate = { slate_id: number; songs: SlateSong[] };
export type Feature = { feature: string; weight: number };
export type Profile = { n_ratings: number; likes: Feature[]; dislikes: Feature[] };
export type Stats = { n_ratings: number; by_context: Record<string, number> };

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    throw new Error(detail.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  contexts: () => call<Contexts>("/api/contexts"),
  stats: () => call<Stats>("/api/stats"),
  profile: () => call<Profile>("/api/profile"),
  startSession: (context: string) =>
    call<{ session_id: number; context: string }>("/api/sessions", {
      method: "POST",
      body: JSON.stringify({ context }),
    }),
  nextSlate: (sessionId: number) =>
    call<Slate>(`/api/sessions/${sessionId}/slates`, { method: "POST" }),
  answer: (slateId: number, chosen: number, worst: number | null) =>
    call<{ n_ratings: number }>(`/api/slates/${slateId}/answer`, {
      method: "POST",
      body: JSON.stringify({ chosen_song_id: chosen, worst_song_id: worst }),
    }),
};

export type RequestCreated = { request_id: number; tags: string[]; energy: string | null };
export type RequestStatus = {
  status: "generating" | "ready" | "failed";
  n_done: number;
  n_failed: number;
  n_total: number;
  slate: Slate | null;
  error: string | null;
};

export const requestsApi = {
  create: (text: string, context: string, referenceId: number | null = null) =>
    call<RequestCreated>("/api/requests", {
      method: "POST",
      body: JSON.stringify({ text, context, reference_id: referenceId }),
    }),
  status: (requestId: number) => call<RequestStatus>(`/api/requests/${requestId}`),
};

export type SimilarSong = { song_id: number; audio_url: string; prompt: string; distance: number };
export type Reference = {
  reference_id: number;
  filename: string;
  duration_s: number;
  status: "pending" | "ready" | "failed";
  error: string | null;
  audio_url: string;
  similar: SimilarSong[]; // closest library songs, once ready
};

export const referencesApi = {
  // The body is the raw file, not a multipart form.
  upload: (file: File) =>
    call<Reference>(`/api/references?filename=${encodeURIComponent(file.name)}`, {
      method: "POST",
      body: file,
      headers: { "Content-Type": file.type || "application/octet-stream" },
    }),
  get: (referenceId: number) => call<Reference>(`/api/references/${referenceId}`),
};
